/// database_manager_test.dart
/// ===========================
/// Unit tests for [DatabaseManager] covering:
///
///   Test 1 — Store round-trip (MediaAsset + FrameEmbedding)
///   Test 2 — KNN basic: closest vector returned as rank-1
///   Test 3 — Hybrid filter: audio (mediaType=2) excluded from visual search
///   Test 4 — TranscriptSegment round-trip (sourceType discrimination)
///
/// Running
/// -------
/// From the `media_core_ffi/` root:
///
///   flutter test test/database_manager_test.dart --reporter expanded
///
/// Prerequisites
/// -------------
/// Run `flutter pub run build_runner build --delete-conflicting-outputs` to
/// generate `lib/objectbox.g.dart` BEFORE running these tests.  The generated
/// file provides [getObjectBoxModel] and the `MediaAsset_`, `FrameEmbedding_`,
/// and `TranscriptSegment_` property accessor classes.
///
/// Store isolation
/// ---------------
/// Each test creates its own temporary directory and tears it down in
/// [tearDown].  Tests are fully isolated — no shared state bleeds between them.
library;

import 'dart:io';
import 'dart:math';

import 'package:flutter_test/flutter_test.dart';
import 'package:objectbox/objectbox.dart';

import 'package:media_core_ffi/objectbox.g.dart';
import 'package:media_core_ffi/database_manager.dart';

// ── Helpers ───────────────────────────────────────────────────────────────────

const _kDimensions = 1152; // SigLIP SO400M — must match @HnswIndex(dimensions:)

/// Builds a unit-norm 1152-dim float vector.
///
/// [seed] controls the [Random] so that different seeds produce meaningfully
/// different vectors (they will NOT be nearest-neighbours of each other when
/// seeds differ by large integers).
List<double> _unitVector(int seed) {
  final rng = Random(seed);
  final raw = List<double>.generate(_kDimensions, (_) => rng.nextDouble() - 0.5);
  final norm = sqrt(raw.fold<double>(0.0, (sum, v) => sum + v * v));
  return norm > 0 ? raw.map((v) => v / norm).toList() : raw;
}

/// Builds an exact copy of [source] with a tiny epsilon perturbation so that
/// cosine similarity is extremely high (but not numerically identical).
List<double> _nearlyIdenticalVector(List<double> source) {
  final perturbed = List<double>.from(source);
  perturbed[0] += 1e-6;
  // Re-normalise
  final norm = sqrt(perturbed.fold<double>(0.0, (s, v) => s + v * v));
  return perturbed.map((v) => v / norm).toList();
}

// ── Test fixture ─────────────────────────────────────────────────────────────

/// Opens a fresh, isolated [Store] in a temporary directory.
///
/// Returns both the [Store] and the temp directory so [tearDown] can delete it.
(Store, Directory) _openTempStore() {
  final tempDir = Directory.systemTemp.createTempSync('objectbox_test_');
  final store = Store(
    getObjectBoxModel(),
    directory: tempDir.path,
  );
  return (store, tempDir);
}

/// Closes [store] and deletes [tempDir].
void _closeTempStore(Store store, Directory tempDir) {
  if (!store.isClosed()) store.close();
  if (tempDir.existsSync()) tempDir.deleteSync(recursive: true);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

void main() {
  // Each test declares its own store/dir locals and manages lifecycle in
  // setUp/tearDown so failures in one test cannot pollute others.

  group('DatabaseManager', () {
    late DatabaseManager db;
    late Store _store;
    late Directory _tempDir;

    setUp(() async {
      (_store, _tempDir) = _openTempStore();
      // Inject the already-opened store via the internal constructor path.
      // We expose a factory for testing so we don't have to hit the filesystem
      // twice via openStore().
      db = DatabaseManager.fromStore(_store);
    });

    tearDown(() {
      db.dispose();
      _closeTempStore(_store, _tempDir);
    });

    // ── Test 1: Store round-trip ─────────────────────────────────────────────

    test('T1 · MediaAsset + FrameEmbedding round-trip', () {
      // Arrange
      final asset = MediaAsset(
        filePath: '/test/videos/sample.mp4',
        mediaType: 1, // Video
        durationMs: 60000,
        dateAddedMs: 1700000000000,
      );
      final assetId = db.putAsset(asset);
      expect(assetId, greaterThan(0), reason: 'putAsset must return a valid ID');

      final embedding = FrameEmbedding(
        timestampMs: 3500,
        visualVector: _unitVector(42),
      )..mediaAsset.targetId = assetId;
      final embId = db.putEmbedding(embedding);
      expect(embId, greaterThan(0), reason: 'putEmbedding must return a valid ID');

      // Act — read back
      final fetchedAsset = db.getAsset(assetId);
      expect(fetchedAsset, isNotNull);
      expect(fetchedAsset!.filePath, equals('/test/videos/sample.mp4'));
      expect(fetchedAsset.mediaType, equals(1));
      expect(fetchedAsset.durationMs, equals(60000));
      expect(fetchedAsset.isFullyIndexed, isFalse);

      // Verify the embedding's parent link
      final embBox = _store.box<FrameEmbedding>();
      final fetchedEmb = embBox.get(embId)!;
      expect(fetchedEmb.timestampMs, equals(3500));
      expect(fetchedEmb.visualVector.length, equals(_kDimensions));
      expect(fetchedEmb.mediaAsset.targetId, equals(assetId));
    });

    // ── Test 2: KNN basic ────────────────────────────────────────────────────

    test('T2 · KNN returns the closest vector as rank-1', () {
      // Arrange — insert 5 known vectors linked to 5 different image assets
      final vectors = List.generate(5, (i) => _unitVector(i * 100));
      for (int i = 0; i < 5; i++) {
        final asset = MediaAsset(
          filePath: '/test/images/img_$i.jpg',
          mediaType: 0, // Image
          durationMs: 0,
          dateAddedMs: 1700000000000 + i,
        );
        final aid = db.putAsset(asset);

        final emb = FrameEmbedding(
          timestampMs: 0,
          visualVector: vectors[i],
        )..mediaAsset.targetId = aid;
        db.putEmbedding(emb);
      }

      // Query with a near-copy of vector[2].
      // maxResults=1 → ObjectBox returns the single nearest neighbour.
      // With k=1, results.first IS the closest vector by definition.
      final queryVec = _nearlyIdenticalVector(vectors[2]);
      final results = db.searchVisualSemantic(
        queryVec,
        maxResults: 1,          // ← key: k=1 so result[0] is unambiguously rank-1
        excludeMediaType: -1,   // no filter — all media types
      );

      expect(results.length, equals(1),
          reason: 'k=1 search must return exactly 1 result');

      final rank1 = results.first;
      expect(rank1.visualVector.length, equals(_kDimensions),
          reason: 'Returned vector must still be 1152-dim');

      // Helper: cosine dot product for unit-norm vectors = cosine similarity
      double dot(List<double> a, List<double> b) =>
          List.generate(a.length, (i) => a[i] * b[i]).fold(0.0, (s, v) => s + v);

      final rank1Score = dot(rank1.visualVector, queryVec);

      // rank-1 must be vectors[2] (≈ identical to queryVec).
      // Cosine similarity of two nearly-identical unit vectors → ≈ 1.0.
      expect(rank1Score, greaterThan(0.999),
          reason: 'k=1 KNN must return vectors[2] (cosine sim ≈ 1.0); '
              'got $rank1Score — HNSW returned the wrong vector');

      // Verify it beats every other stored vector against the query.
      for (int i = 0; i < 5; i++) {
        final simToRank1 = dot(vectors[i], rank1.visualVector);
        if (simToRank1 < 0.999) {
          // vectors[i] is a genuinely different vector
          final otherScore = dot(vectors[i], queryVec);
          expect(rank1Score, greaterThan(otherScore),
              reason: 'rank-1 (score=$rank1Score) must beat vector[$i] '
                  '(score=$otherScore)');
        }
      }
    });

    // ── Test 3: Hybrid filter ────────────────────────────────────────────────

    test('T3 · Hybrid filter excludes audio (mediaType=2) from visual search', () {
      // Arrange — 3 video assets + 2 audio assets, all with similar embeddings
      final baseVec = _unitVector(999);

      final videoIds = <int>[];
      for (int i = 0; i < 3; i++) {
        final asset = MediaAsset(
          filePath: '/test/videos/v$i.mp4',
          mediaType: 1, // Video
          durationMs: 30000,
          dateAddedMs: 1700000000000 + i,
        );
        videoIds.add(db.putAsset(asset));
      }

      final audioIds = <int>[];
      for (int i = 0; i < 2; i++) {
        final asset = MediaAsset(
          filePath: '/test/audio/a$i.mp3',
          mediaType: 2, // Audio — should be excluded
          durationMs: 180000,
          dateAddedMs: 1700000000000 + 10 + i,
        );
        audioIds.add(db.putAsset(asset));
      }

      // Give each asset a near-identical embedding so all 5 would rank highly
      for (final aid in [...videoIds, ...audioIds]) {
        final emb = FrameEmbedding(
          timestampMs: 0,
          visualVector: _nearlyIdenticalVector(baseVec),
        )..mediaAsset.targetId = aid;
        db.putEmbedding(emb);
      }

      // Act — search with default excludeMediaType=2 (audio excluded)
      final results = db.searchVisualSemantic(baseVec, maxResults: 10);

      expect(results, isNotEmpty, reason: 'Should return video results');
      expect(
        results.length,
        lessThanOrEqualTo(3),
        reason: 'At most 3 results (only the 3 video assets)',
      );

      // Assert — no result links back to an audio asset
      for (final r in results) {
        final parentId = r.mediaAsset.targetId;
        expect(
          audioIds.contains(parentId),
          isFalse,
          reason: 'Audio asset id=$parentId must not appear in visual search results',
        );
        expect(
          videoIds.contains(parentId),
          isTrue,
          reason: 'Only video asset IDs should appear in results',
        );
      }
    });

    // ── Test 4: TranscriptSegment round-trip ─────────────────────────────────

    test('T4 · TranscriptSegment round-trip with sourceType discrimination', () {
      // Arrange
      final asset = MediaAsset(
        filePath: '/test/videos/lecture.mp4',
        mediaType: 1,
        durationMs: 300000,
        dateAddedMs: 1700000000000,
      );
      final assetId = db.putAsset(asset);

      // One Whisper ASR segment
      final whisper = TranscriptSegment(
        text: 'Hello and welcome to the lecture.',
        startTimeMs: 1000,
        endTimeMs: 4500,
        sourceType: 0, // Whisper
      )..mediaAsset.targetId = assetId;

      // One PP-OCR segment
      final ocr = TranscriptSegment(
        text: 'CHAPTER 1',
        startTimeMs: 5000,
        endTimeMs: 8000,
        sourceType: 1, // PP-OCR
      )..mediaAsset.targetId = assetId;

      db.putTranscripts([whisper, ocr]);

      // Act
      final fetched = db.getTranscriptsForAsset(assetId);

      // Assert — both segments present, correct sourceType values
      expect(fetched.length, equals(2));

      final whisperSegs = fetched.where((s) => s.sourceType == 0).toList();
      final ocrSegs = fetched.where((s) => s.sourceType == 1).toList();

      expect(whisperSegs.length, equals(1));
      expect(whisperSegs.first.text, equals('Hello and welcome to the lecture.'));
      expect(whisperSegs.first.startTimeMs, equals(1000));
      expect(whisperSegs.first.endTimeMs, equals(4500));

      expect(ocrSegs.length, equals(1));
      expect(ocrSegs.first.text, equals('CHAPTER 1'));
      expect(ocrSegs.first.startTimeMs, equals(5000));
      expect(ocrSegs.first.endTimeMs, equals(8000));
    });
  });
}

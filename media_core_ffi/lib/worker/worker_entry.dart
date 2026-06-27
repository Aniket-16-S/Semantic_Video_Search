import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:convert';
import 'dart:async';
import 'package:flutter/services.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';
import 'package:media_core_ffi/database_manager.dart';
import 'package:media_core_ffi/media_core_ffi.dart';

import '../models/media_asset.dart';
import '../models/frame_embedding.dart';
import '../models/transcript_segment.dart';
import 'messages.dart';
import '../inference/siglip_runner.dart';
import '../inference/whisper_runner.dart';
import '../inference/ocr_runner.dart';

/// Loads model bytes. Tries File I/O first for local paths, falls back to rootBundle
/// if the models are bundled as Flutter assets.
Future<Uint8List> _loadModelBytes(String path) async {
  final file = File(path);
  if (file.existsSync()) {
    return file.readAsBytesSync();
  } else {
    final byteData = await rootBundle.load(path);
    return byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes);
  }
}

/// The main entry point for the persistent background worker isolate.
void workerEntry(WorkerBootstrap bootstrap) async {
  final receivePort = ReceivePort();
  final mainPort = bootstrap.sendPort;

  try {
    // Initialize isolate context for platform channels (needed for rootBundle)
    if (bootstrap.rootIsolateToken != null) {
      BackgroundIsolateBinaryMessenger.ensureInitialized(bootstrap.rootIsolateToken!);
    }

    // Initialize ONNX Runtime Environment
    OrtEnv.instance.init();

    // Establish bidirectional comms
    mainPort.send(WorkerHandshake(receivePort.sendPort));

    // Load models sequentially to manage memory spikes
    final siglipVisionBytes = await _loadModelBytes(bootstrap.models.siglipVisionPath);
    final siglipRunner = await SigLipRunner.load(siglipVisionBytes);

    final whisperBytes = await _loadModelBytes(bootstrap.models.whisperPath);
    final whisperRunner = await WhisperRunner.load(whisperBytes);

    final ocrDetBytes = await _loadModelBytes(bootstrap.models.ocrDetPath);
    final ocrRecBytes = await _loadModelBytes(bootstrap.models.ocrRecPath);
    final ocrRunner = await OcrRunner.load(detBytes: ocrDetBytes, recBytes: ocrRecBytes);

    // Open ObjectBox storage
    final db = DatabaseManager();
    db.initFromReference(bootstrap.storeReference);

    // Notify UI that we are fully booted and ready
    mainPort.send(WorkerReady());

    // Main event loop
    await for (final message in receivePort) {
      if (message is IndexRequest) {
        print('--- BACKGROUND WORKER: Started indexing ${message.filePath} ---');
        mainPort.send(IndexProgress(message.filePath, 0, 1));
        
        int embeddingsWritten = 0;
        int transcriptSegmentsWritten = 0;
        
        try {
          // ── Deduplication Guard ─────────────────────────────────────────────
          // Check if this file already exists in the database.
          final existing = db.findAssetByPath(message.filePath);
          
          if (existing != null && existing.isFullyIndexed) {
            // Already fully processed — skip entirely to prevent duplicates.
            print('[WORKER] Skipping already-indexed: ${message.filePath}');
            mainPort.send(IndexResult(
              filePath: message.filePath,
              success: true,
              embeddingsWritten: 0,
              transcriptSegmentsWritten: 0,
            ));
            print('--- BACKGROUND WORKER: Finished indexing ${message.filePath} ---');
            continue;
          }
          
          // If partially indexed (app crashed mid-way), reuse existing ID so
          // we don't create orphaned duplicate rows.
          final int assetId;
          if (existing != null) {
            assetId = existing.id;
            print('[WORKER] Resuming partial indexing for: ${message.filePath}');
          } else {
            final asset = MediaAsset(
              filePath: message.filePath, 
              mediaType: message.mediaType, 
              durationMs: 0,
              dateAddedMs: DateTime.now().millisecondsSinceEpoch,
              isFullyIndexed: false,
            );
            assetId = db.putAsset(asset);
          }
          // ───────────────────────────────────────────────────────────────────
          
          if (message.mediaType == 1 || message.mediaType == 2) {
            final pcmBytes = MediaCore.extractAudio(message.filePath);
            if (pcmBytes != null) {
               final segments = whisperRunner.transcribeChunk(pcmBytes);
               for (var seg in segments) {
                 seg.mediaAsset.targetId = assetId;
                 db.putTranscript(seg);
                 transcriptSegmentsWritten++;
               }
            }
          }

          if (message.mediaType == 0 || message.mediaType == 1) {
             final frames = MediaCore.extractVideoFrames(message.filePath);
             if (frames != null) {
               int frameIdx = 0;
               for (final frame in frames) {
                 final Float32List float32List = frame['data'];
                 final double timestamp = frame['timestamp_s'];

                 final emb = siglipRunner.encodeFrame(float32List);
                 final frameEmbedding = FrameEmbedding(
                    timestampMs: (timestamp * 1000).toInt(),
                    visualVector: emb,
                 );
                 frameEmbedding.mediaAsset.targetId = assetId;
                 db.putEmbedding(frameEmbedding);
                 embeddingsWritten++;

                 // Route the normalized CHW Float32List directly into the OCR pipeline natively
                 final ocrResults = ocrRunner.detectText(float32List);
                 if (ocrResults.isNotEmpty) {
                   final ocrText = ocrResults.map((e) => e.text).join(" ");
                   final seg = TranscriptSegment(
                     text: ocrText,
                     startTimeMs: (timestamp * 1000).toInt(),
                     endTimeMs: (timestamp * 1000).toInt() + 1000,
                     sourceType: 1,
                   );
                   seg.mediaAsset.targetId = assetId;
                   db.putTranscript(seg);
                   transcriptSegmentsWritten++;
                 }

                 frameIdx++;
                 mainPort.send(IndexProgress(message.filePath, frameIdx, frames.length));
               }
             }
          }
          
          // Mark the asset as fully indexed so future launches skip it.
          final indexedAsset = db.getAsset(assetId);
          if (indexedAsset != null) {
            indexedAsset.isFullyIndexed = true;
            db.putAsset(indexedAsset);
          }
          
          mainPort.send(IndexResult(
            filePath: message.filePath,
            success: true,
            embeddingsWritten: embeddingsWritten,
            transcriptSegmentsWritten: transcriptSegmentsWritten,
          ));
        } catch (e, st) {
          print('Error during indexing: $e\n$st');
          mainPort.send(IndexResult(
            filePath: message.filePath,
            success: false,
            embeddingsWritten: embeddingsWritten,
            transcriptSegmentsWritten: transcriptSegmentsWritten,
            errorMessage: e.toString(),
          ));
        }
        
        print('--- BACKGROUND WORKER: Finished indexing ${message.filePath} ---');
      } else if (message is ShutdownCommand) {
        break; // Exit the loop to tear down
      }
    }

    // Teardown sequence
    siglipRunner.dispose();
    whisperRunner.dispose();
    ocrRunner.dispose();
    db.dispose();
    OrtEnv.instance.release();
    Isolate.current.kill();
  } catch (e, st) {
    mainPort.send(WorkerError('$e\n$st'));
  }
}

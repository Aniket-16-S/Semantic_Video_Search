import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';

// ── Models ───────────────────────────────────────────────────────────────────

class ModelPaths {
  final String siglipVisionPath;
  final String siglipTextPath;
  final String whisperPath;
  final String ocrDetPath;
  final String ocrRecPath;

  const ModelPaths({
    required this.siglipVisionPath,
    required this.siglipTextPath,
    required this.whisperPath,
    required this.ocrDetPath,
    required this.ocrRecPath,
  });
}

// ── Internal Handshake / Bootstrap ──────────────────────────────────────────

class WorkerBootstrap {
  final ByteData storeReference;
  final ModelPaths models;
  final SendPort sendPort;
  final RootIsolateToken? rootIsolateToken;

  const WorkerBootstrap({
    required this.storeReference,
    required this.models,
    required this.sendPort,
    this.rootIsolateToken,
  });
}

class WorkerHandshake {
  final SendPort sendPort;
  const WorkerHandshake(this.sendPort);
}

// ── UI -> Worker Commands ────────────────────────────────────────────────────

sealed class WorkerCommand {}

class IndexRequest extends WorkerCommand {
  final String filePath;
  final int mediaType; // 0=Image, 1=Video, 2=Audio

  IndexRequest({
    required this.filePath,
    required this.mediaType,
  });
}

class ShutdownCommand extends WorkerCommand {}

// ── Worker -> UI Events ──────────────────────────────────────────────────────

sealed class WorkerEvent {}

class IndexProgress extends WorkerEvent {
  final String filePath;
  final int framesProcessed;
  final int framesTotal;

  IndexProgress(this.filePath, this.framesProcessed, this.framesTotal);
}

class IndexResult extends WorkerEvent {
  final String filePath;
  final bool success;
  final String? errorMessage;
  final int embeddingsWritten;
  final int transcriptSegmentsWritten;

  IndexResult({
    required this.filePath,
    required this.success,
    this.errorMessage,
    this.embeddingsWritten = 0,
    this.transcriptSegmentsWritten = 0,
  });
}

class WorkerReady extends WorkerEvent {}

class WorkerError extends WorkerEvent {
  final String message;
  WorkerError(this.message);
}

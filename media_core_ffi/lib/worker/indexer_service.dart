import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';

import 'messages.dart';
import 'worker_entry.dart';

/// The public API for the main UI isolate to interact with the background
/// processing worker.
class IndexerService {
  final SendPort _sendPort;
  final ReceivePort _receivePort;
  final StreamController<WorkerEvent> _eventStreamController = StreamController.broadcast();
  final Isolate _isolate;
  bool _isShuttingDown = false;

  IndexerService._(this._sendPort, this._receivePort, this._isolate);

  /// Spawns the worker isolate, loads ONNX models, and completes the handshake.
  static Future<IndexerService> create({
    required ByteData storeReference,
    required ModelPaths models,
  }) async {
    final receivePort = ReceivePort();
    final rootToken = RootIsolateToken.instance;

    final bootstrap = WorkerBootstrap(
      storeReference: storeReference,
      models: models,
      sendPort: receivePort.sendPort,
      rootIsolateToken: rootToken,
    );

    final isolate = await Isolate.spawn(workerEntry, bootstrap);

    final completer = Completer<IndexerService>();
    SendPort? workerSendPort;
    IndexerService? service;

    receivePort.listen((message) {
      if (message is WorkerHandshake) {
        workerSendPort = message.sendPort;
      } else if (message is WorkerReady) {
        service = IndexerService._(workerSendPort!, receivePort, isolate);
        completer.complete(service);
      } else if (message is WorkerEvent) {
        if (service != null && !service!._isShuttingDown) {
          service!._eventStreamController.add(message);
        } else if (message is WorkerError && !completer.isCompleted) {
          completer.completeError(Exception('Worker failed to start: ${message.message}'));
        }
      }
    });

    return completer.future;
  }

  /// Stream of events (progress, results, errors) from the worker.
  Stream<WorkerEvent> get events => _eventStreamController.stream;

  /// Submit a file for indexing. Returns immediately.
  void submit(IndexRequest request) {
    if (_isShuttingDown) throw StateError('IndexerService is shutting down');
    _sendPort.send(request);
  }

  /// Request the worker to shutdown gracefully.
  Future<void> dispose() async {
    if (_isShuttingDown) return;
    _isShuttingDown = true;
    
    _sendPort.send(ShutdownCommand());
    await _eventStreamController.close();
    _receivePort.close();
    _isolate.kill();
  }
}

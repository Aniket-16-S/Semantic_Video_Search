import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'database_manager.dart';
import 'state/gallery_provider.dart';
import 'ui/gallery_screen.dart';
import 'worker/indexer_service.dart';
import 'worker/messages.dart';
import 'services/media_scanner_service.dart';

import 'dart:io';
import 'package:path_provider/path_provider.dart';

void main() async {
  // Ensure native bindings are ready
  WidgetsFlutterBinding.ensureInitialized();

  // Create ObjectBox directory safely
  final appDir = await getApplicationSupportDirectory();
  final dbDir = Directory('${appDir.path}/objectbox');
  if (!dbDir.existsSync()) {
    dbDir.createSync(recursive: true);
  }

  // Boot ObjectBox
  final db = DatabaseManager();
  await db.init(directory: dbDir.path);

  // Boot Background AI Worker
  final indexer = await IndexerService.create(
    storeReference: db.store.reference,
    models: const ModelPaths(
      siglipVisionPath: 'assets/models/vision_encoder_int8.onnx',
      siglipTextPath: 'assets/models/text_encoder_int8.onnx',
      whisperPath: 'assets/models/whisper_tiny_int8/encoder_model.onnx',
      ocrDetPath: 'assets/models/ppocr_det_int8.onnx',
      ocrRecPath: 'assets/models/ppocr_rec_int8.onnx',
    ),
  );

  // Initialize Media Scanner and trigger background scan
  final scanner = MediaScannerService(indexerService: indexer);
  scanner.scanDevice();

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => GalleryProvider(db)),
      ],
      child: MaterialApp(
        title: 'Semantic Video Search',
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.cyan,
            brightness: Brightness.dark,
          ),
          appBarTheme: const AppBarTheme(
            backgroundColor: Colors.transparent,
            elevation: 0,
          ),
        ),
        home: const GalleryScreen(),
        debugShowCheckedModeBanner: false,
      ),
    ),
  );
}

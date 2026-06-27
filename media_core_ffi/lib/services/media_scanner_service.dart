import 'dart:io';
import 'dart:isolate';
import 'package:photo_manager/photo_manager.dart';
import 'package:path_provider/path_provider.dart';

import '../worker/indexer_service.dart';
import '../worker/messages.dart';

class MediaScannerService {
  final IndexerService indexerService;

  MediaScannerService({required this.indexerService});

  /// Master function to trigger the scan based on the current OS
  Future<void> scanDevice() async {
    if (Platform.isAndroid) {
      await _scanAndroid();
    } else if (Platform.isWindows) {
      await _scanWindows();
    }
  }

  /// ANDROID LOGIC (Using photo_manager)
  Future<void> _scanAndroid() async {
    // 1. Request OS Permissions
    final PermissionState ps = await PhotoManager.requestPermissionExtend();
    if (!ps.isAuth) return; // Handle denied permissions in UI later

    // 2. Fetch all media albums
    List<AssetPathEntity> albums = await PhotoManager.getAssetPathList(
      type: RequestType.common, // Includes Image, Video, Audio
    );

    if (albums.isEmpty) return;

    // 3. Get the "Recent" album (usually the first one)
    List<AssetEntity> recentMedia = await albums[0].getAssetListPaged(page: 0, size: 100);

    for (AssetEntity asset in recentMedia) {
      // Get the raw local file path
      File? file = await asset.file; 
      if (file != null) {
        // Send the path to our C++/ONNX Isolate for embedding!
        _dispatchToWorker(file.path, _mapPhotoManagerTypeToInt(asset.typeInt));
      }
    }
  }

  /// WINDOWS LOGIC (Using path_provider + dart:io)
  Future<void> _scanWindows() async {
    // 1. Get standard Windows directories
    final Directory picturesDir = await getApplicationDocumentsDirectory(); // Expand to other dirs as needed
    
    // 2. Simple recursive scan for common media extensions
    final List<String> targetExtensions = ['.jpg', '.png', '.mp4', '.mp3'];
    
    await for (FileSystemEntity entity in picturesDir.list(recursive: true, followLinks: false)) {
      if (entity is File) {
        String ext = entity.path.split('.').last.toLowerCase();
        if (targetExtensions.contains('.$ext')) {
           // Send the path to our C++/ONNX Isolate for embedding!
           _dispatchToWorker(entity.path, _mapExtensionToTypeInt(ext));
        }
      }
    }
  }

  /// Dispatcher Hook
  void _dispatchToWorker(String filePath, int mediaType) {
    // We send an IndexRequest to our IndexerService
    indexerService.submit(IndexRequest(
      filePath: filePath,
      mediaType: mediaType,
    ));
    print("Dispatched to Background Engine: $filePath");
  }

  int _mapPhotoManagerTypeToInt(int typeInt) {
    // photo_manager: 1=Image, 2=Video, 3=Audio
    // our app: 0=Image, 1=Video, 2=Audio
    if (typeInt == 1) return 0; // Image
    if (typeInt == 2) return 1; // Video
    if (typeInt == 3) return 2; // Audio
    return 0; // Default to Image
  }

  int _mapExtensionToTypeInt(String ext) {
    if (['jpg', 'png'].contains(ext)) return 0; // Image
    if (ext == 'mp4') return 1; // Video
    if (ext == 'mp3') return 2; // Audio
    return 0;
  }
}

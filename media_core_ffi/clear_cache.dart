import 'dart:io';

void main() {
  print('Locating ObjectBox cache directory...');
  // For Windows, default path provider puts it in Documents
  final envDocs = Platform.environment['USERPROFILE'];
  if (envDocs != null) {
    // path_provider puts it here:
    final docsDir = Directory('$envDocs\\Documents\\media_core_ffi');
    if (docsDir.existsSync()) {
      print('Found cache at: ${docsDir.path}');
      try {
        docsDir.deleteSync(recursive: true);
        print('✅ Successfully cleared the database cache from Documents!');
      } catch (e) {
        print('Error deleting cache: $e');
      }
      return;
    }
  }
  
  // Also check local directory just in case
  final localDir = Directory('objectbox');
  if (localDir.existsSync()) {
    print('Found local cache at: ${localDir.path}');
    try {
      localDir.deleteSync(recursive: true);
      print('✅ Successfully cleared the local database cache!');
    } catch (e) {
      print('Error deleting cache: $e');
    }
    return;
  }
  
  print('No cache found! You are good to go.');
}

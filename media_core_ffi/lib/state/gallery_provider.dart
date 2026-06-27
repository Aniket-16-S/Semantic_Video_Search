import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:media_core_ffi/database_manager.dart';

/// State Management layer for the Gallery UI.
/// Listens to ObjectBox reactive streams and pushes updates to the widgets.
class GalleryProvider extends ChangeNotifier {
  final DatabaseManager _db;
  StreamSubscription<List<MediaAsset>>? _assetSubscription;

  // Normal Gallery State
  List<MediaAsset> _assets = [];
  List<MediaAsset> get assets => _assets;

  // Semantic Search State
  bool _isSearching = false;
  bool get isSearching => _isSearching;

  List<FrameEmbedding> _searchResults = [];
  List<FrameEmbedding> get searchResults => _searchResults;

  GalleryProvider(this._db) {
    _initStreams();
  }

  void _initStreams() {
    // Watch the ObjectBox database for changes to MediaAssets.
    // Whenever the Worker Isolate writes a new asset, this stream emits the
    // updated list and we notify the UI to rebuild the GridView.
    _assetSubscription = _db.watchAllAssets().listen((updatedAssets) {
      _assets = updatedAssets;
      notifyListeners();
    });
  }

  /// Triggers a semantic search using the pre-encoded text query vector
  void performSearch(List<double> queryVector, {int excludeMediaType = 2}) {
    _isSearching = true;
    _searchResults = _db.searchVisualSemantic(
      queryVector,
      maxResults: 50,
      excludeMediaType: excludeMediaType, // Default ignores Audio files
    );
    notifyListeners();
  }

  /// Clears the semantic search and returns to the normal gallery view
  void clearSearch() {
    _isSearching = false;
    _searchResults.clear();
    notifyListeners();
  }

  @override
  void dispose() {
    _assetSubscription?.cancel();
    super.dispose();
  }
}

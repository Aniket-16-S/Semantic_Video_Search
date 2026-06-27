import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:video_thumbnail/video_thumbnail.dart';

import '../state/gallery_provider.dart';
import '../database_manager.dart';
import 'widgets/search_bar_widget.dart';

class GalleryScreen extends StatelessWidget {
  const GalleryScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0D0D0D),
      extendBodyBehindAppBar: true,
      appBar: PreferredSize(
        preferredSize: const Size.fromHeight(kToolbarHeight),
        child: ClipRRect(
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: AppBar(
              title: const Text(
                'AI Gallery',
                style: TextStyle(
                  fontWeight: FontWeight.w600,
                  letterSpacing: 1.1,
                  color: Colors.white,
                ),
              ),
              backgroundColor: const Color(0xFF1A1A1A).withValues(alpha: 0.6),
              elevation: 0,
              actions: [
                FuturisticSearchBar(
                  onSearch: (query, excludeMediaType) {
                    // TODO: Replace dummy vector with actual SigLipTextRunner output
                    final dummyVector = List<double>.filled(1152, 0.001);
                    Provider.of<GalleryProvider>(context, listen: false)
                        .performSearch(dummyVector, excludeMediaType: excludeMediaType);
                  },
                  onClear: () {
                    Provider.of<GalleryProvider>(context, listen: false).clearSearch();
                  },
                ),
                const SizedBox(width: 8),
              ],
            ),
          ),
        ),
      ),
      body: Consumer<GalleryProvider>(
        builder: (context, provider, child) {
          final assets = provider.isSearching ? <MediaAsset>[] : provider.assets;
          final searchResults = provider.searchResults;

          if (provider.isSearching) {
            if (searchResults.isEmpty) {
              return const Center(
                child: Text('No semantic matches found', style: TextStyle(color: Colors.white54)),
              );
            }
            return _buildSearchGrid(context, searchResults);
          }

          if (assets.isEmpty) {
            return const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.photo_library_outlined, size: 64, color: Colors.white54),
                  SizedBox(height: 16),
                  Text(
                    'No Media Found',
                    style: TextStyle(color: Colors.white54, fontSize: 18),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Indexing in background...',
                    style: TextStyle(color: Colors.white38, fontSize: 14),
                  ),
                ],
              ),
            );
          }

          return GridView.builder(
            padding: EdgeInsets.only(
              top: MediaQuery.paddingOf(context).top + kToolbarHeight + 8,
              left: 4,
              right: 4,
              bottom: MediaQuery.paddingOf(context).bottom + 8,
            ),
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
              crossAxisSpacing: 4,
              mainAxisSpacing: 4,
              childAspectRatio: 1,
            ),
            itemCount: assets.length,
            itemBuilder: (context, index) {
              return MediaThumbnail(asset: assets[index]);
            },
          );
        },
      ),
    );
  }

  Widget _buildSearchGrid(BuildContext context, List<FrameEmbedding> searchResults) {
    return GridView.builder(
      padding: EdgeInsets.only(
        top: MediaQuery.paddingOf(context).top + kToolbarHeight + 8,
        left: 4,
        right: 4,
        bottom: MediaQuery.paddingOf(context).bottom + 8,
      ),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 3,
        crossAxisSpacing: 4,
        mainAxisSpacing: 4,
        childAspectRatio: 1,
      ),
      itemCount: searchResults.length,
      itemBuilder: (context, index) {
        final hit = searchResults[index];
        final asset = hit.mediaAsset.target;
        if (asset == null) return const SizedBox.shrink();

        return Stack(
          fit: StackFit.expand,
          children: [
            MediaThumbnail(asset: asset),
            if (asset.mediaType == 1)
              Positioned(
                top: 4,
                left: 4,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.cyanAccent.withValues(alpha: 0.8),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '${(hit.timestampMs / 1000).toStringAsFixed(1)}s',
                    style: const TextStyle(
                        color: Colors.black, fontSize: 10, fontWeight: FontWeight.bold),
                  ),
                ),
              ),
          ],
        );
      },
    );
  }
}

// ── MediaThumbnail ─────────────────────────────────────────────────────────────

class MediaThumbnail extends StatelessWidget {
  final MediaAsset asset;

  const MediaThumbnail({Key? key, required this.asset}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    IconData? mediaIcon;
    if (asset.mediaType == 1) mediaIcon = Icons.play_circle_fill;
    if (asset.mediaType == 2) mediaIcon = Icons.audiotrack;

    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Background Image / Video Thumbnail
          _buildImageLayer(),

          // Gradient overlay for icon visibility
          if (mediaIcon != null)
            Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.transparent, Colors.black54],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
              ),
            ),

          // Media type icon / summarize menu
          if (mediaIcon != null)
            Positioned(
              bottom: 6,
              right: 6,
              child: asset.mediaType == 1
                  ? _buildSummarizationMenu(context, mediaIcon)
                  : Icon(mediaIcon, color: Colors.white, size: 24),
            ),

          // Indexing spinner if not yet fully processed
          if (!asset.isFullyIndexed)
            const Positioned(
              top: 6,
              right: 6,
              child: SizedBox(
                width: 12,
                height: 12,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white70),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildSummarizationMenu(BuildContext context, IconData icon) {
    return PopupMenuButton<String>(
      icon: Icon(icon, color: Colors.white, size: 24),
      color: const Color(0xFF2A2A2A),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      tooltip: 'Summarize Video',
      onSelected: (value) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Starting $value summarization...'),
            backgroundColor: Colors.cyan[800],
          ),
        );
      },
      itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
        const PopupMenuItem<String>(
          value: 'Frames Only',
          child: Text('Frames Only (Visual)', style: TextStyle(color: Colors.white)),
        ),
        const PopupMenuItem<String>(
          value: 'Audio Only',
          child: Text('Audio Only (Transcribe)', style: TextStyle(color: Colors.white)),
        ),
        const PopupMenuItem<String>(
          value: 'Full Summary',
          child: Text('Full Summary (Both)', style: TextStyle(color: Colors.white)),
        ),
      ],
    );
  }

  Widget _buildImageLayer() {
    // ── Audio: styled gradient ───────────────────────────────────────────────
    if (asset.mediaType == 2) {
      return Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [Color(0xFF2C3E50), Color(0xFF000000)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
      );
    }

    // ── Video: generate a real frame thumbnail via video_thumbnail ───────────
    if (asset.mediaType == 1) {
      return FutureBuilder<Uint8List?>(
        key: ValueKey(asset.filePath),
        future: VideoThumbnail.thumbnailData(
          video: asset.filePath,
          imageFormat: ImageFormat.JPEG,
          maxWidth: 300,
          quality: 75,
        ),
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const _VideoLoadingPlaceholder();
          }
          if (snapshot.hasData && snapshot.data != null) {
            return Image.memory(
              snapshot.data!,
              fit: BoxFit.cover,
            );
          }
          // Failed to extract thumbnail — show styled fallback with filename
          return _VideoFallbackPlaceholder(filePath: asset.filePath);
        },
      );
    }

    // ── Image: standard file render ─────────────────────────────────────────
    final file = File(asset.filePath);
    if (!file.existsSync()) {
      return Container(color: Colors.grey[900]);
    }

    return Image.file(
      file,
      fit: BoxFit.cover,
      cacheWidth: 300,
      errorBuilder: (context, error, stackTrace) {
        return Container(
          color: Colors.grey[900],
          child: const Icon(Icons.broken_image, color: Colors.white38),
        );
      },
    );
  }
}

// ── Video thumbnail loading placeholder (animated shimmer) ─────────────────────

class _VideoLoadingPlaceholder extends StatefulWidget {
  const _VideoLoadingPlaceholder();

  @override
  State<_VideoLoadingPlaceholder> createState() => _VideoLoadingPlaceholderState();
}

class _VideoLoadingPlaceholderState extends State<_VideoLoadingPlaceholder>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;
  late final Animation<double> _anim;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
    _anim = CurvedAnimation(parent: _ctrl, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _anim,
      builder: (_, __) => Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color(0xFF1A1A2E),
              Color.lerp(const Color(0xFF16213E), const Color(0xFF0F3460), _anim.value)!,
              const Color(0xFF1A1A2E),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        child: const Center(
          child: Icon(Icons.videocam, color: Colors.white24, size: 28),
        ),
      ),
    );
  }
}

// ── Video thumbnail fallback (extraction failed) ──────────────────────────────

class _VideoFallbackPlaceholder extends StatelessWidget {
  final String filePath;
  const _VideoFallbackPlaceholder({required this.filePath});

  @override
  Widget build(BuildContext context) {
    final name = filePath.split(Platform.pathSeparator).last;
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [Color(0xFF0F3460), Color(0xFF16213E)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
      ),
      padding: const EdgeInsets.all(8),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.movie_outlined, color: Colors.cyanAccent, size: 28),
          const SizedBox(height: 6),
          Text(
            name,
            style: const TextStyle(color: Colors.white54, fontSize: 9),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}

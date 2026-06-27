import 'dart:ui';
import 'package:flutter/material.dart';

class FuturisticSearchBar extends StatefulWidget {
  final Function(String query, int excludeMediaType) onSearch;
  final VoidCallback onClear;

  const FuturisticSearchBar({
    Key? key,
    required this.onSearch,
    required this.onClear,
  }) : super(key: key);

  @override
  State<FuturisticSearchBar> createState() => _FuturisticSearchBarState();
}

class _FuturisticSearchBarState extends State<FuturisticSearchBar>
    with SingleTickerProviderStateMixin {
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();

  bool _isExpanded = false;
  int _excludeMediaType = 2; // 2 = Audio (default, visual search)

  late AnimationController _animController;
  late Animation<double> _expandAnimation;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 300));
    _expandAnimation =
        CurvedAnimation(parent: _animController, curve: Curves.easeInOutBack);
  }

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    _animController.dispose();
    super.dispose();
  }

  void _toggleSearch() {
    setState(() {
      _isExpanded = !_isExpanded;
      if (_isExpanded) {
        _animController.forward();
        _focusNode.requestFocus();
      } else {
        _animController.reverse();
        _focusNode.unfocus();
        _controller.clear();
        widget.onClear();
      }
    });
  }

  void _submitSearch() {
    if (_controller.text.trim().isNotEmpty) {
      widget.onSearch(_controller.text.trim(), _excludeMediaType);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeOutExpo,
      width: _isExpanded ? MediaQuery.sizeOf(context).width - 32 : 48,
      height: _isExpanded ? 200 : 48,
      decoration: BoxDecoration(
        color: _isExpanded
            ? const Color(0xFF1E1E1E).withValues(alpha: 0.9)
            : Colors.transparent,
        borderRadius: BorderRadius.circular(24),
        border:
            _isExpanded ? Border.all(color: Colors.white24, width: 1) : null,
        boxShadow: _isExpanded
            ? [
                BoxShadow(
                    color: Colors.cyanAccent.withValues(alpha: 0.2),
                    blurRadius: 20,
                    spreadRadius: -5)
              ]
            : [],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(24),
        child: BackdropFilter(
          filter: ImageFilter.blur(
              sigmaX: _isExpanded ? 15 : 0, sigmaY: _isExpanded ? 15 : 0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Expanded(
                    child: _isExpanded
                        ? Padding(
                            padding: const EdgeInsets.only(left: 20),
                            child: TextField(
                              controller: _controller,
                              focusNode: _focusNode,
                              style: const TextStyle(
                                  color: Colors.white, fontSize: 16),
                              decoration: const InputDecoration(
                                hintText:
                                    'Search visually (e.g. "dog playing in park")',
                                hintStyle: TextStyle(color: Colors.white54),
                                border: InputBorder.none,
                              ),
                              onSubmitted: (_) => _submitSearch(),
                            ),
                          )
                        : const SizedBox.shrink(),
                  ),
                  IconButton(
                    icon: Icon(_isExpanded ? Icons.close : Icons.search),
                    color: _isExpanded ? Colors.white54 : Colors.white,
                    onPressed: _toggleSearch,
                  ),
                ],
              ),
              if (_isExpanded)
                SizeTransition(
                  sizeFactor: _expandAnimation,
                  child: Padding(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                    child: SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: [
                          const Text('Exclude:',
                              style:
                                  TextStyle(color: Colors.white54, fontSize: 12)),
                          const SizedBox(width: 12),
                          _buildFilterChip('Audio', 2),
                          const SizedBox(width: 8),
                          _buildFilterChip('Images', 0),
                          const SizedBox(width: 8),
                          _buildFilterChip('Videos', 1),
                          const SizedBox(width: 16),
                          ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor:
                                  Colors.cyanAccent.withValues(alpha: 0.8),
                              foregroundColor: Colors.black,
                              shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20)),
                            ),
                            onPressed: _submitSearch,
                            child: const Text('Search',
                                style: TextStyle(fontWeight: FontWeight.bold)),
                          )
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFilterChip(String label, int mediaType) {
    final isSelected = _excludeMediaType == mediaType;
    return GestureDetector(
      onTap: () {
        setState(() {
          // Toggle off if already selected (means exclude nothing, i.e., -1)
          _excludeMediaType = isSelected ? -1 : mediaType;
        });
      },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected
              ? Colors.redAccent.withValues(alpha: 0.3)
              : Colors.white10,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
              color: isSelected ? Colors.redAccent : Colors.transparent),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.redAccent : Colors.white54,
            fontSize: 12,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class LocationPickerService {
  static const MethodChannel _channel = MethodChannel('com.example.crop_wise/location_picker');
  
  static Future<void> showLocationPicker() async {
    try {
      await _channel.invokeMethod('showLocationPicker');
    } on PlatformException catch (e) {
      print('Error showing location picker: ${e.message}');
    }
  }
  
  static void setLocationPickerCallbacks({
    required Function(double latitude, double longitude) onLocationSelected,
    required VoidCallback onLocationPickerDismissed,
  }) {
    _channel.setMethodCallHandler((call) async {
      switch (call.method) {
        case 'onLocationSelected':
          final Map<String, dynamic> locationData = Map<String, dynamic>.from(call.arguments);
          onLocationSelected(
            locationData['latitude'],
            locationData['longitude'],
          );
          break;
        case 'onLocationPickerDismissed':
          onLocationPickerDismissed();
          break;
      }
    });
  }
}

class CustomLocationPicker extends StatefulWidget {
  final Function(double latitude, double longitude) onLocationSelected;
  final VoidCallback? onLocationPickerDismissed;

  const CustomLocationPicker({
    Key? key,
    required this.onLocationSelected,
    this.onLocationPickerDismissed,
  }) : super(key: key);

  @override
  State<CustomLocationPicker> createState() => _CustomLocationPickerState();
}

class _CustomLocationPickerState extends State<CustomLocationPicker> {
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    LocationPickerService.setLocationPickerCallbacks(
      onLocationSelected: (latitude, longitude) {
        setState(() => _isLoading = false);
        widget.onLocationSelected(latitude, longitude);
      },
      onLocationPickerDismissed: () {
        setState(() => _isLoading = false);
        widget.onLocationPickerDismissed?.call();
      },
    );
  }

  Future<void> _openLocationPicker() async {
    setState(() => _isLoading = true);
    await LocationPickerService.showLocationPicker();
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Select Location Manually',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: _isLoading ? null : _openLocationPicker,
            icon: _isLoading 
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.map),
            label: Text(_isLoading ? 'Opening Map...' : 'Open Map Picker'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 12),
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}

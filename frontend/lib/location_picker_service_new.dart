import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class LocationPickerServiceNew {
  static const MethodChannel _channel = MethodChannel('crop_wise/location_picker');

  static Future<Map<String, dynamic>?> showLocationPicker() async {
    try {
      final result = await _channel.invokeMethod('showLocationPicker');
      
      if (result != null && result is Map) {
        return Map<String, dynamic>.from(result);
      }
      return null;
    } on PlatformException catch (e) {
      debugPrint('Error showing location picker: ${e.message}');
      return null;
    } catch (e) {
      debugPrint('Unexpected error: $e');
      return null;
    }
  }
}

class LocationPickerWidgetNew extends StatelessWidget {
  final double? latitude;
  final double? longitude;
  final Function(double, double, Map<String, dynamic>) onLocationSelected;

  const LocationPickerWidgetNew({
    Key? key,
    this.latitude,
    this.longitude,
    required this.onLocationSelected,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 300,
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: Colors.grey.shade300),
        borderRadius: BorderRadius.circular(12),
        color: Colors.white,
      ),
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.green.shade50,
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(12),
                topRight: Radius.circular(12),
              ),
            ),
            child: Row(
              children: [
                Icon(Icons.location_on, color: Colors.green.shade700),
                const SizedBox(width: 8),
                Text(
                  'Manual Location Selection',
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    color: Colors.green.shade700,
                  ),
                ),
              ],
            ),
          ),
          
          // Current location display or placeholder
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(16),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (latitude != null && longitude != null)
                    Column(
                      children: [
                        Icon(Icons.location_on, size: 48, color: Colors.green.shade600),
                        const SizedBox(height: 8),
                        Text(
                          'Current Location',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey.shade600,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '${latitude!.toStringAsFixed(6)}, ${longitude!.toStringAsFixed(6)}',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w500,
                            fontFamily: 'monospace',
                          ),
                        ),
                      ],
                    )
                  else
                    Column(
                      children: [
                        Icon(Icons.location_searching, size: 48, color: Colors.grey.shade400),
                        const SizedBox(height: 8),
                        Text(
                          'No location selected',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey.shade600,
                          ),
                        ),
                      ],
                    ),
                ],
              ),
            ),
          ),
          
          // Action button
          Container(
            padding: const EdgeInsets.all(16),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton.icon(
                onPressed: () async {
                  final result = await LocationPickerServiceNew.showLocationPicker();
                  if (result != null) {
                    final lat = result['latitude'] as double;
                    final lng = result['longitude'] as double;
                    final locationData = result['locationData'] as Map<String, dynamic>;
                    onLocationSelected(lat, lng, locationData);
                  }
                },
                icon: const Icon(Icons.map),
                label: const Text('Open Map Picker'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green.shade600,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

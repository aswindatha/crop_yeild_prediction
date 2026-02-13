import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:http/http.dart' as http;
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      systemNavigationBarColor: Color(0xFFF8FAF6),
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );
  runApp(const CropYieldApp());
}

// Design tokens
class AppColors {
  static const primary = Color(0xFF2D6A4F);
  static const primaryLight = Color(0xFF40916C);
  static const primaryDark = Color(0xFF1B4332);
  static const accent = Color(0xFF95D5B2);
  static const surface = Color(0xFFF8FAF6);
  static const cardBg = Color(0xFFFFFFFF);
  static const textPrimary = Color(0xFF1B4332);
  static const textSecondary = Color(0xFF52796F);
  static const success = Color(0xFF2D6A4F);
  static const error = Color(0xFFB71C1C);
}

class CropYieldApp extends StatelessWidget {
  const CropYieldApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CropWise',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: AppColors.primary,
          primary: AppColors.primary,
          surface: AppColors.surface,
          brightness: Brightness.light,
        ),
        scaffoldBackgroundColor: AppColors.surface,
        textTheme: GoogleFonts.poppinsTextTheme(),
        appBarTheme: AppBarTheme(
          elevation: 0,
          centerTitle: true,
          titleTextStyle: GoogleFonts.poppins(
            fontSize: 20,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
          iconTheme: const IconThemeData(color: Colors.white),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          color: AppColors.cardBg,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: BorderSide(color: Colors.grey.shade200),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: AppColors.primary, width: 2),
          ),
          contentPadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          hintStyle: GoogleFonts.poppins(color: Colors.grey.shade500),
        ),
      ),
      home: const PredictionScreen(),
    );
  }
}

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  // Form controllers
  final TextEditingController _phosphorusController = TextEditingController();
  final TextEditingController _potassiumController = TextEditingController();
  final TextEditingController _farmSizeController = TextEditingController();
  final TextEditingController _apiUrlController = TextEditingController();
  
  // Form state
  String? _selectedCropType;
  bool _irrigationAvailable = false;
  Position? _currentPosition;
  bool _isLoading = false;
  bool _isTestingConnection = false;
  String _connectionStatus = '';
  bool _connectionSuccessful = false;
  String _apiUrl = 'http://172.16.208.211:5000'; // Using the current IP address from your Flask server logs
  
  // Crop types dropdown
  final List<String> _cropTypes = [
    'rice',
    'sugarcane', 
    'cotton',
    'pulses',
    'millet',
    'groundnut',
    'coconut'
  ];

  @override
  void initState() {
    super.initState();
    _getCurrentLocation();
    _loadApiUrl();
  }

  Future<void> _loadApiUrl() async {
    // In a real app, you'd load this from shared preferences
    _apiUrlController.text = _apiUrl;
  }

  Future<void> _getCurrentLocation() async {
    try {
      // Check location permission
      var status = await Permission.location.request();
      if (status.isGranted) {
        Position position = await Geolocator.getCurrentPosition(
          desiredAccuracy: LocationAccuracy.high,
        );
        setState(() {
          _currentPosition = position;
        });
      } else {
        _showMessage('Location permission denied. Please enable location services.');
      }
    } catch (e) {
      _showMessage('Error getting location: $e');
    }
  }

  Future<void> _testConnection() async {
    setState(() {
      _isTestingConnection = true;
      _connectionStatus = 'Testing connection...';
      _connectionSuccessful = false;
    });

    try {
      final testUrl = _apiUrlController.text.trim();
      final response = await http.get(
        Uri.parse('$testUrl/health'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 10));

      setState(() {
        if (response.statusCode == 200) {
          _connectionStatus = '✅ Connection successful!';
          _connectionSuccessful = true;
        } else {
          _connectionStatus = '❌ Server error: ${response.statusCode}';
          _connectionSuccessful = false;
        }
      });
    } catch (e) {
      setState(() {
        _connectionStatus = '❌ Connection failed: $e';
        _connectionSuccessful = false;
      });
    } finally {
      setState(() {
        _isTestingConnection = false;
      });
    }
  }

  Future<void> _saveApiUrl() async {
    final url = _apiUrlController.text.trim();
    if (url.isNotEmpty) {
      setState(() {
        _apiUrl = url;
      });
      // In a real app, save to shared preferences
      Navigator.of(context).pop();
      _showMessage('API URL saved successfully!');
    }
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return _SettingsDialog(
          apiUrlController: _apiUrlController,
          onSave: (String url) {
            setState(() {
              _apiUrl = url;
            });
            Navigator.of(context).pop();
            _showMessage('API URL saved successfully!');
          },
        );
      },
    );
  }

  Future<void> _predictYield() async {
    if (_currentPosition == null) {
      _showMessage('Please enable location services');
      return;
    }

    if (_selectedCropType == null) {
      _showMessage('Please select a crop type');
      return;
    }

    if (_phosphorusController.text.isEmpty ||
        _potassiumController.text.isEmpty ||
        _farmSizeController.text.isEmpty) {
      _showMessage('Please fill all fields');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final response = await http.post(
        Uri.parse('$_apiUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'latitude': _currentPosition!.latitude,
          'longitude': _currentPosition!.longitude,
          'crop_type': _selectedCropType,
          'phosphorus': double.parse(_phosphorusController.text),
          'potassium': double.parse(_potassiumController.text),
          'irrigation_available': _irrigationAvailable ? 1 : 0,
          'farm_size_ha': double.parse(_farmSizeController.text),
        }),
      ).timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final result = jsonDecode(response.body);
        _showPredictionResult(result);
      } else {
        _showMessage('Prediction failed: ${response.statusCode}');
      }
    } catch (e) {
      _showMessage('Error: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showPredictionResult(Map<String, dynamic> result) {
    final yieldKg = (result['predicted_yield_kg'] as num).toDouble();
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    gradient: LinearGradient(colors: [AppColors.primary.withOpacity(0.15), AppColors.accent.withOpacity(0.2)], begin: Alignment.topLeft, end: Alignment.bottomRight),
                    shape: BoxShape.circle,
                  ),
                  child: Image.asset('assets/images/logo.png', width: 48, height: 48),
                  // child: Icon(Icons.eco_rounded, size: 48, color: AppColors.primary),
                ),
                const SizedBox(height: 20),
                Text('Predicted Yield', style: GoogleFonts.poppins(fontSize: 15, color: AppColors.textSecondary, fontWeight: FontWeight.w500)),
                const SizedBox(height: 6),
                Text('${yieldKg.toStringAsFixed(1)} kg', style: GoogleFonts.poppins(fontSize: 32, fontWeight: FontWeight.bold, color: AppColors.primary)),
                const SizedBox(height: 20),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(color: Colors.grey.shade50, borderRadius: BorderRadius.circular(14)),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _resultRow('Location', result['api_data']?['location']?['name']?.toString() ?? 'Unknown Location'),
                      _resultRow('Crop', (result['input_features']['crop_type'] as String).toString()),
                      _resultRow('Region', (result['api_data']['region'] as String).toString()),
                      _resultRow('Temperature', '${result['api_data']['weather']['avg_temp']}°C'),
                      _resultRow('Rainfall', '${result['api_data']['weather']['rainfall']} mm'),
                      _resultRow('Soil pH', result['api_data']['soil']['phh2o'].toString()),
                      _resultRow('Farm Size', '${result['input_features']['farm_size_ha']} ha'),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () => Navigator.of(context).pop(),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    ),
                    child: Text('Done', style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _resultRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: GoogleFonts.poppins(color: AppColors.textSecondary, fontSize: 13)),
          Text(value, style: GoogleFonts.poppins(fontWeight: FontWeight.w500, color: AppColors.textPrimary)),
        ],
      ),
    );
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: GoogleFonts.poppins()),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
        backgroundColor: AppColors.textPrimary,
      ),
    );
  }

  Widget _sectionCard({
    required String title,
    required IconData icon,
    required Color iconBg,
    required Widget child,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.cardBg,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: iconBg.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(icon, color: iconBg, size: 22),
                ),
                const SizedBox(width: 12),
                Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontSize: 17,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            child,
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        slivers: [
          SliverAppBar(
            expandedHeight: 120,
            floating: true,
            pinned: true,
            backgroundColor: AppColors.primary,
            flexibleSpace: Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [AppColors.primaryDark, AppColors.primary, AppColors.primaryLight],
                ),
              ),
            ),
            title: Text(
              'CropWise',
              style: GoogleFonts.poppins(
                fontSize: 20,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            actions: [
              IconButton(
                icon: const Icon(Icons.settings_rounded),
                onPressed: _showSettingsDialog,
              ),
            ],
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 28),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Location
                  _sectionCard(
                    title: 'Location',
                    icon: Icons.location_on_rounded,
                    iconBg: AppColors.primary,
                    child: _currentPosition != null
                        ? Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              _infoRow(Icons.my_location_rounded, 'Lat', _currentPosition!.latitude.toStringAsFixed(6)),
                              const SizedBox(height: 8),
                              _infoRow(Icons.my_location_rounded, 'Lon', _currentPosition!.longitude.toStringAsFixed(6)),
                              const SizedBox(height: 12),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                                decoration: BoxDecoration(
                                  color: AppColors.success.withOpacity(0.12),
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: Row(
                                  children: [
                                    Icon(Icons.check_circle_rounded, size: 18, color: AppColors.success),
                                    const SizedBox(width: 8),
                                    Text('Location acquired', style: GoogleFonts.poppins(fontSize: 13, color: AppColors.success, fontWeight: FontWeight.w500)),
                                  ],
                                ),
                              ),
                            ],
                          )
                        : Row(
                            children: [
                              SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primary),
                              ),
                              const SizedBox(width: 12),
                              Text('Getting location...', style: GoogleFonts.poppins(color: AppColors.textSecondary)),
                            ],
                          ),
                  ),
                  const SizedBox(height: 18),

                  // Crop Type
                  _sectionCard(
                    title: 'Crop Type',
                    icon: Icons.eco_rounded,
                    iconBg: const Color(0xFF40916C),
                    child: DropdownButtonFormField<String>(
                      value: _selectedCropType,
                      decoration: InputDecoration(
                        hintText: 'Select crop',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(14)),
                        filled: true,
                        fillColor: Colors.grey.shade50,
                      ),
                      items: _cropTypes.map((crop) => DropdownMenuItem(value: crop, child: Text(crop.capitalize(), style: GoogleFonts.poppins()))).toList(),
                      onChanged: (v) => setState(() => _selectedCropType = v),
                    ),
                  ),
                  const SizedBox(height: 18),

                  // Nutrients
                  _sectionCard(
                    title: 'Nutrients',
                    icon: Icons.science_rounded,
                    iconBg: const Color(0xFF2C7A7B),
                    child: Column(
                      children: [
                        TextField(
                          controller: _phosphorusController,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(labelText: 'Phosphorus (mg/kg)', hintText: 'e.g. 25.0'),
                        ),
                        const SizedBox(height: 14),
                        TextField(
                          controller: _potassiumController,
                          keyboardType: TextInputType.number,
                          decoration: const InputDecoration(labelText: 'Potassium (mg/kg)', hintText: 'e.g. 150.0'),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 18),

                  // Irrigation
                  _sectionCard(
                    title: 'Irrigation',
                    icon: Icons.water_drop_rounded,
                    iconBg: const Color(0xFF0077B6),
                    child: Row(
                      children: [
                        Expanded(
                          child: _choiceChip(
                            label: 'Yes',
                            selected: _irrigationAvailable,
                            onTap: () => setState(() => _irrigationAvailable = true),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _choiceChip(
                            label: 'No',
                            selected: !_irrigationAvailable,
                            onTap: () => setState(() => _irrigationAvailable = false),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 18),

                  // Farm Size
                  _sectionCard(
                    title: 'Farm Size',
                    icon: Icons.landscape_rounded,
                    iconBg: const Color(0xFF6A4C93),
                    child: TextField(
                      controller: _farmSizeController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(labelText: 'Area (hectares)', hintText: 'e.g. 3.0'),
                    ),
                  ),
                  const SizedBox(height: 28),

                  // CTA Button
                  Material(
                    color: Colors.transparent,
                    child: InkWell(
                      onTap: _isLoading ? null : _predictYield,
                      borderRadius: BorderRadius.circular(18),
                      child: Container(
                        height: 58,
                        decoration: BoxDecoration(
                          gradient: _isLoading ? null : const LinearGradient(colors: [AppColors.primaryDark, AppColors.primary], begin: Alignment.topLeft, end: Alignment.bottomRight),
                          color: _isLoading ? Colors.grey.shade300 : null,
                          borderRadius: BorderRadius.circular(18),
                          boxShadow: _isLoading ? null : [BoxShadow(color: AppColors.primary.withOpacity(0.35), blurRadius: 16, offset: const Offset(0, 8))],
                        ),
                        child: Center(
                          child: _isLoading
                              ? const SizedBox(width: 26, height: 26, child: CircularProgressIndicator(strokeWidth: 2.5, color: AppColors.primary))
                              : Text('Predict Yield', style: GoogleFonts.poppins(fontSize: 18, fontWeight: FontWeight.w600, color: Colors.white)),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _infoRow(IconData icon, String label, String value) {
    return Row(
      children: [
        Icon(icon, size: 18, color: AppColors.textSecondary),
        const SizedBox(width: 8),
        Text('$label: ', style: GoogleFonts.poppins(color: AppColors.textSecondary, fontSize: 14)),
        Text(value, style: GoogleFonts.poppins(fontWeight: FontWeight.w500, color: AppColors.textPrimary)),
      ],
    );
  }

  Widget _choiceChip({required String label, required bool selected, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: selected ? AppColors.primary.withOpacity(0.12) : Colors.grey.shade100,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: selected ? AppColors.primary : Colors.transparent, width: 2),
          ),
          child: Center(
            child: Text(label, style: GoogleFonts.poppins(fontWeight: selected ? FontWeight.w600 : FontWeight.w500, color: selected ? AppColors.primary : AppColors.textSecondary)),
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _phosphorusController.dispose();
    _potassiumController.dispose();
    _farmSizeController.dispose();
    _apiUrlController.dispose();
    super.dispose();
  }
}

class _SettingsDialog extends StatefulWidget {
  final TextEditingController apiUrlController;
  final Function(String) onSave;

  const _SettingsDialog({
    required this.apiUrlController,
    required this.onSave,
  });

  @override
  State<_SettingsDialog> createState() => _SettingsDialogState();
}

class _SettingsDialogState extends State<_SettingsDialog> {
  bool _isTestingConnection = false;
  String _connectionStatus = '';
  bool _connectionSuccessful = false;

  Future<void> _testConnection() async {
    setState(() {
      _isTestingConnection = true;
      _connectionStatus = 'Testing connection...';
      _connectionSuccessful = false;
    });

    try {
      final testUrl = widget.apiUrlController.text.trim();
      final response = await http.get(
        Uri.parse('$testUrl/health'),
        headers: {'Content-Type': 'application/json'},
      ).timeout(const Duration(seconds: 10));

      setState(() {
        if (response.statusCode == 200) {
          _connectionStatus = '✅ Connection successful!';
          _connectionSuccessful = true;
        } else {
          _connectionStatus = '❌ Server error: ${response.statusCode}';
          _connectionSuccessful = false;
        }
        _isTestingConnection = false;
      });
    } catch (e) {
      setState(() {
        _connectionStatus = '❌ Connection failed: $e';
        _connectionSuccessful = false;
        _isTestingConnection = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(color: AppColors.primary.withOpacity(0.12), borderRadius: BorderRadius.circular(12)),
                  child: Icon(Icons.settings_rounded, color: AppColors.primary, size: 22),
                ),
                const SizedBox(width: 12),
                Text('Settings', style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
              ],
            ),
            const SizedBox(height: 20),
            Text('Backend API URL', style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.w500, color: AppColors.textSecondary)),
            const SizedBox(height: 8),
            TextField(
              controller: widget.apiUrlController,
              decoration: const InputDecoration(hintText: 'http://172.16.208.211:5000'),
            ),
            const SizedBox(height: 16),
            if (_connectionStatus.isNotEmpty)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: _connectionSuccessful ? AppColors.success.withOpacity(0.1) : AppColors.error.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: _connectionSuccessful ? AppColors.success.withOpacity(0.3) : AppColors.error.withOpacity(0.3)),
                ),
                child: Row(
                  children: [
                    Icon(_connectionSuccessful ? Icons.check_circle_rounded : Icons.error_rounded, size: 20, color: _connectionSuccessful ? AppColors.success : AppColors.error),
                    const SizedBox(width: 10),
                    Expanded(child: Text(_connectionStatus, style: GoogleFonts.poppins(fontSize: 13, color: _connectionSuccessful ? AppColors.success : AppColors.error))),
                  ],
                ),
              ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _isTestingConnection ? null : _testConnection,
                    icon: _isTestingConnection ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.wifi_tethering_rounded, size: 18),
                    label: Text(_isTestingConnection ? 'Testing...' : 'Test', style: GoogleFonts.poppins(fontWeight: FontWeight.w500)),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppColors.primary,
                      side: const BorderSide(color: AppColors.primary),
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: _connectionSuccessful
                        ? () {
                            final url = widget.apiUrlController.text.trim();
                            if (url.isNotEmpty) widget.onSave(url);
                          }
                        : null,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: Colors.grey.shade300,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                    ),
                    child: Text('Save', style: GoogleFonts.poppins(fontWeight: FontWeight.w600)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: Text('Cancel', style: GoogleFonts.poppins(color: AppColors.textSecondary)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

extension StringExtension on String {
  String capitalize() {
    return "${this[0].toUpperCase()}${substring(1).toLowerCase()}";
  }
}

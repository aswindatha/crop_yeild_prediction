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

class _PredictionScreenState extends State<PredictionScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  
  // Form controllers
  final TextEditingController _phosphorusController = TextEditingController();
  final TextEditingController _potassiumController = TextEditingController();
  final TextEditingController _farmSizeController = TextEditingController();
  final TextEditingController _apiUrlController = TextEditingController();
  
  // Soil data controllers
  final TextEditingController _soilPhController = TextEditingController();
  final TextEditingController _soilNitrogenController = TextEditingController();
  final TextEditingController _soilOrganicCarbonController = TextEditingController();
  final TextEditingController _soilSandController = TextEditingController();
  final TextEditingController _soilSiltController = TextEditingController();
  final TextEditingController _soilClayController = TextEditingController();
  final TextEditingController _soilCecController = TextEditingController();
  
  // Synthetic soil nutrient controllers (for display only)
  final TextEditingController _ammoniaController = TextEditingController();
  final TextEditingController _nitrateController = TextEditingController();
  final TextEditingController _zincController = TextEditingController();
  final TextEditingController _ironController = TextEditingController();
  final TextEditingController _manganeseController = TextEditingController();
  
  // Form state
  String? _selectedCropType;
  bool _irrigationAvailable = false;
  Position? _currentPosition;
  double? _currentAccuracy;
  String? _locationTimestamp;
  bool _isLoading = false;
  bool _isTestingConnection = false;
  String _connectionStatus = '';
  bool _connectionSuccessful = false;
  String _apiUrl = 'http://172.16.208.211:5000'; // Using the current IP address from your Flask server logs
  
  // Validation errors
  Map<String, String> _validationErrors = {};
  
  // High-precision location tracking
  final List<Position> _locationReadings = [];
  bool _isGettingHighPrecisionLocation = false;
  
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
    _tabController = TabController(length: 2, vsync: this);
    _getCurrentLocation();
    _loadApiUrl();
    
    // Add listeners to clear validation errors when user starts typing
    _phosphorusController.addListener(_clearValidationErrors);
    _potassiumController.addListener(_clearValidationErrors);
    _farmSizeController.addListener(_clearValidationErrors);
    _soilPhController.addListener(_clearValidationErrors);
    _soilNitrogenController.addListener(_clearValidationErrors);
    _soilOrganicCarbonController.addListener(_clearValidationErrors);
    _soilSandController.addListener(_clearValidationErrors);
    _soilSiltController.addListener(_clearValidationErrors);
    _soilClayController.addListener(_clearValidationErrors);
    _soilCecController.addListener(_clearValidationErrors);
    
    // Synthetic nutrient listeners (clear validation when user types)
    _ammoniaController.addListener(_clearValidationErrors);
    _nitrateController.addListener(_clearValidationErrors);
    _zincController.addListener(_clearValidationErrors);
    _ironController.addListener(_clearValidationErrors);
    _manganeseController.addListener(_clearValidationErrors);
  }

  Future<void> _loadApiUrl() async {
    // In a real app, you'd load this from shared preferences
    _apiUrlController.text = _apiUrl;
  }

  // High-precision location methods
  Future<void> _getCurrentLocation() async {
    try {
      setState(() {
        _isGettingHighPrecisionLocation = true;
        _locationReadings.clear();
      });
      
      // Check location permission
      var status = await Permission.location.request();
      if (!status.isGranted) {
        _showMessage('Location permission denied. Please enable location services.');
        setState(() {
          _isGettingHighPrecisionLocation = false;
        });
        return;
      }
      
      // Collect 5 high-precision readings
      for (int i = 0; i < 5; i++) {
        try {
          Position position = await Geolocator.getCurrentPosition(
            desiredAccuracy: LocationAccuracy.bestForNavigation,
            timeLimit: const Duration(seconds: 15),
          );
          
          _locationReadings.add(position);
          
          // Wait a bit between readings for better accuracy
          if (i < 4) {
            await Future.delayed(const Duration(milliseconds: 500));
          }
        } catch (e) {
          print('Error getting reading ${i + 1}: $e');
        }
      }
      
      if (_locationReadings.isEmpty) {
        _showMessage('Failed to get any location readings');
        setState(() {
          _isGettingHighPrecisionLocation = false;
        });
        return;
      }
      
      // Calculate averaged position and accuracy
      final avgLat = _locationReadings.map((p) => p.latitude).reduce((a, b) => a + b) / _locationReadings.length;
      final avgLon = _locationReadings.map((p) => p.longitude).reduce((a, b) => a + b) / _locationReadings.length;
      final avgAccuracy = _locationReadings.map((p) => p.accuracy).reduce((a, b) => a + b) / _locationReadings.length;
      
      // Filter by accuracy (only use if < 50 meters)
      if (avgAccuracy > 50) {
        _showMessage('Location accuracy too poor (${avgAccuracy.toStringAsFixed(1)}m). Please try again in an open area.');
        setState(() {
          _isGettingHighPrecisionLocation = false;
        });
        return;
      }
      
      setState(() {
        _currentPosition = Position(
          latitude: avgLat,
          longitude: avgLon,
          accuracy: avgAccuracy,
          altitude: _locationReadings.first.altitude,
          altitudeAccuracy: _locationReadings.first.altitudeAccuracy,
          heading: _locationReadings.first.heading,
          headingAccuracy: _locationReadings.first.headingAccuracy ?? 0.0,
          speed: _locationReadings.first.speed,
          speedAccuracy: _locationReadings.first.speedAccuracy ?? 0.0,
          timestamp: _locationReadings.first.timestamp,
        );
        _currentAccuracy = avgAccuracy;
        _locationTimestamp = DateTime.now().toIso8601String();
        _isGettingHighPrecisionLocation = false;
      });
      
      _showMessage('High-precision location acquired (${avgAccuracy.toStringAsFixed(1)}m accuracy)');
      
    } catch (e) {
      _showMessage('Error getting location: $e');
      setState(() {
        _isGettingHighPrecisionLocation = false;
      });
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

  // Validation methods
  bool _validateBasicTab() {
    final errors = <String, String>{};
    
    // Validate phosphorus (mg/kg) - typical range: 0-100 mg/kg
    final phosphorus = double.tryParse(_phosphorusController.text);
    if (phosphorus == null) {
      errors['phosphorus'] = 'Please enter a valid phosphorus value';
    } else if (phosphorus < 0 || phosphorus > 100) {
      errors['phosphorus'] = 'Phosphorus must be between 0 and 100 mg/kg';
    }
    
    // Validate potassium (mg/kg) - typical range: 0-300 mg/kg
    final potassium = double.tryParse(_potassiumController.text);
    if (potassium == null) {
      errors['potassium'] = 'Please enter a valid potassium value';
    } else if (potassium < 0 || potassium > 300) {
      errors['potassium'] = 'Potassium must be between 0 and 300 mg/kg';
    }
    
    // Validate farm size (hectares) - typical range: 0.1-1000 hectares
    final farmSize = double.tryParse(_farmSizeController.text);
    if (farmSize == null) {
      errors['farm_size'] = 'Please enter a valid farm size';
    } else if (farmSize < 0.1 || farmSize > 1000) {
      errors['farm_size'] = 'Farm size must be between 0.1 and 1000 hectares';
    }
    
    setState(() {
      _validationErrors = errors;
    });
    
    return errors.isEmpty;
  }
  
  bool _validateSoilApiTab() {
    final errors = <String, String>{};
    
    // Validate soil pH - typical range: 3.0-10.0
    final soilPh = double.tryParse(_soilPhController.text);
    if (soilPh == null) {
      errors['soil_ph'] = 'Please enter a valid soil pH value';
    } else if (soilPh < 3.0 || soilPh > 10.0) {
      errors['soil_ph'] = 'Soil pH must be between 3.0 and 10.0';
    }
    
    // Validate nitrogen (mg/kg) - typical range: 0-200 mg/kg
    final nitrogen = double.tryParse(_soilNitrogenController.text);
    if (nitrogen == null) {
      errors['soil_nitrogen'] = 'Please enter a valid nitrogen value';
    } else if (nitrogen < 0 || nitrogen > 200) {
      errors['soil_nitrogen'] = 'Nitrogen must be between 0 and 200 mg/kg';
    }
    
    // Validate organic carbon (%) - typical range: 0-10%
    final organicCarbon = double.tryParse(_soilOrganicCarbonController.text);
    if (organicCarbon == null) {
      errors['soil_organic_carbon'] = 'Please enter a valid organic carbon value';
    } else if (organicCarbon < 0 || organicCarbon > 10) {
      errors['soil_organic_carbon'] = 'Organic carbon must be between 0 and 10%';
    }
    
    // Validate sand content (%) - range: 0-100%
    final sand = double.tryParse(_soilSandController.text);
    if (sand == null) {
      errors['soil_sand'] = 'Please enter a valid sand content value';
    } else if (sand < 0 || sand > 100) {
      errors['soil_sand'] = 'Sand content must be between 0 and 100%';
    }
    
    // Validate silt content (%) - range: 0-100%
    final silt = double.tryParse(_soilSiltController.text);
    if (silt == null) {
      errors['soil_silt'] = 'Please enter a valid silt content value';
    } else if (silt < 0 || silt > 100) {
      errors['soil_silt'] = 'Silt content must be between 0 and 100%';
    }
    
    // Validate clay content (%) - range: 0-100%
    final clay = double.tryParse(_soilClayController.text);
    if (clay == null) {
      errors['soil_clay'] = 'Please enter a valid clay content value';
    } else if (clay < 0 || clay > 100) {
      errors['soil_clay'] = 'Clay content must be between 0 and 100%';
    }
    
    // Validate texture percentages sum to 100%
    if (sand != null && silt != null && clay != null) {
      final total = sand + silt + clay;
      if ((total - 100).abs() > 1.0) { // Allow 1% tolerance
        errors['soil_texture'] = 'Sand + Silt + Clay must equal approximately 100% (current: ${total.toStringAsFixed(1)}%)';
      }
    }
    
    // Validate CEC (optional) - typical range: 1-100 meq/100g
    if (_soilCecController.text.isNotEmpty) {
      final cec = double.tryParse(_soilCecController.text);
      if (cec == null) {
        errors['soil_cec'] = 'Please enter a valid CEC value';
      } else if (cec < 1 || cec > 100) {
        errors['soil_cec'] = 'CEC must be between 1 and 100 meq/100g';
      }
    }
    
    // Validate farm size (hectares) - typical range: 0.1-1000 hectares
    final farmSize = double.tryParse(_farmSizeController.text);
    if (farmSize == null) {
      errors['farm_size'] = 'Please enter a valid farm size';
    } else if (farmSize < 0.1 || farmSize > 1000) {
      errors['farm_size'] = 'Farm size must be between 0.1 and 1000 hectares';
    }
    
    setState(() {
      _validationErrors = errors;
    });
    
    return errors.isEmpty;
  }
  
  void _clearValidationErrors() {
    setState(() {
      _validationErrors.clear();
    });
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

    // Validate based on current tab
    bool isValid;
    if (_tabController.index == 0) {
      isValid = _validateBasicTab();
    } else {
      isValid = _validateSoilApiTab();
    }
    
    if (!isValid) {
      _showMessage('Please correct the validation errors');
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final requestBody = {
        'latitude': _currentPosition!.latitude,
        'longitude': _currentPosition!.longitude,
        'accuracy': _currentAccuracy ?? _currentPosition!.accuracy,
        'timestamp': _locationTimestamp ?? DateTime.now().toIso8601String(),
        'crop_type': _selectedCropType,
        'irrigation_available': _irrigationAvailable ? 1 : 0,
        'farm_size_ha': double.parse(_farmSizeController.text),
      };

      // Add different data based on tab
      if (_tabController.index == 0) {
        // Main tab - use phosphorus and potassium
        requestBody['phosphorus'] = double.parse(_phosphorusController.text);
        requestBody['potassium'] = double.parse(_potassiumController.text);
      } else {
        // Soil API tab - use comprehensive soil data
        requestBody['soil_ph'] = double.parse(_soilPhController.text);
        requestBody['soil_nitrogen'] = double.parse(_soilNitrogenController.text);
        requestBody['soil_organic_carbon'] = double.parse(_soilOrganicCarbonController.text);
        requestBody['soil_sand'] = double.parse(_soilSandController.text);
        requestBody['soil_silt'] = double.parse(_soilSiltController.text);
        requestBody['soil_clay'] = double.parse(_soilClayController.text);
        if (_soilCecController.text.isNotEmpty) {
          requestBody['soil_cec'] = double.parse(_soilCecController.text);
        }
      }

      final response = await http.post(
        Uri.parse('$_apiUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestBody),
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
    
    // Populate synthetic nutrient fields if available
    if (result['api_data']?['synthetic_nutrients'] != null) {
      final nutrients = result['api_data']['synthetic_nutrients'];
      _ammoniaController.text = nutrients['ammonia_mg_kg'].toString();
      _nitrateController.text = nutrients['nitrate_mg_kg'].toString();
      _zincController.text = nutrients['zinc_mg_kg'].toString();
      _ironController.text = nutrients['iron_mg_kg'].toString();
      _manganeseController.text = nutrients['manganese_mg_kg'].toString();
    }
    
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return Dialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          child: Container(
            width: MediaQuery.of(context).size.width * 0.9,
            height: MediaQuery.of(context).size.height * 0.8,
            child: PredictionResultDialog(result: result, yieldKg: yieldKg),
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
            bottom: TabBar(
              controller: _tabController,
              indicatorColor: Colors.white,
              labelColor: Colors.white,
              unselectedLabelColor: Colors.white70,
              labelStyle: GoogleFonts.poppins(fontWeight: FontWeight.w600),
              tabs: const [
                Tab(text: 'Basic'),
                Tab(text: 'Custom Input'),
              ],
            ),
          ),
          SliverFillRemaining(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildBasicTab(),
                _buildSoilApiTab(),
              ],
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

  Widget _buildValidatedTextField({
    required TextEditingController controller,
    required String labelText,
    required String hintText,
    required String fieldKey,
    TextInputType? keyboardType,
    bool readOnly = false,
  }) {
    final hasError = _validationErrors.containsKey(fieldKey);
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      readOnly: readOnly,
      decoration: InputDecoration(
        labelText: labelText,
        hintText: hintText,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: hasError ? const BorderSide(color: AppColors.error, width: 2) : BorderSide(color: Colors.grey.shade300),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: hasError ? const BorderSide(color: AppColors.error, width: 2) : BorderSide(color: Colors.grey.shade200),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: hasError ? const BorderSide(color: AppColors.error, width: 2) : const BorderSide(color: AppColors.primary, width: 2),
        ),
        errorText: hasError ? _validationErrors[fieldKey] : null,
        errorStyle: GoogleFonts.poppins(color: AppColors.error, fontSize: 12),
      ),
    );
  }
  Widget _buildBasicTab() {
    return SingleChildScrollView(
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
                            Text('Location acquired (${_currentAccuracy?.toStringAsFixed(1)}m accuracy)', style: GoogleFonts.poppins(fontSize: 13, color: AppColors.success, fontWeight: FontWeight.w500)),
                          ],
                        ),
                      ),
                    ],
                  )
                : Row(
                    children: [
                      if (_isGettingHighPrecisionLocation) ...[
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primary),
                        ),
                        const SizedBox(width: 12),
                        Text('Getting high-precision location...', style: GoogleFonts.poppins(color: AppColors.textSecondary)),
                      ] else ...[
                        Icon(Icons.location_disabled_rounded, color: AppColors.textSecondary),
                        const SizedBox(width: 12),
                        Text('Location not available', style: GoogleFonts.poppins(color: AppColors.textSecondary)),
                      ],
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
                _buildValidatedTextField(
                  controller: _phosphorusController,
                  labelText: 'Phosphorus (mg/kg)',
                  hintText: 'e.g. 25.0',
                  fieldKey: 'phosphorus',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _potassiumController,
                  labelText: 'Potassium (mg/kg)',
                  hintText: 'e.g. 150.0',
                  fieldKey: 'potassium',
                  keyboardType: TextInputType.number,
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
            child: _buildValidatedTextField(
              controller: _farmSizeController,
              labelText: 'Area (hectares)',
              hintText: 'e.g. 3.0',
              fieldKey: 'farm_size',
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(height: 28),

          // Validation Error Summary
          if (_validationErrors.isNotEmpty && _tabController.index == 0)
            _buildValidationErrorSummary(),
          
          if (_validationErrors.isNotEmpty && _tabController.index == 0)
            const SizedBox(height: 18),

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
    );
  }

  Widget _buildValidationErrorSummary() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.error.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.error.withOpacity(0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.error_outline_rounded, size: 20, color: AppColors.error),
              const SizedBox(width: 8),
              Text('Validation Errors', style: GoogleFonts.poppins(fontWeight: FontWeight.w600, color: AppColors.error)),
            ],
          ),
          const SizedBox(height: 12),
          ..._validationErrors.entries.map((entry) => Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.circle, size: 6, color: AppColors.error),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    entry.value,
                    style: GoogleFonts.poppins(fontSize: 13, color: AppColors.error),
                  ),
                ),
              ],
            ),
          )),
        ],
      ),
    );
  }

  Widget _buildSoilApiTab() {
    return SingleChildScrollView(
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
                            Text('Location acquired (${_currentAccuracy?.toStringAsFixed(1)}m accuracy)', style: GoogleFonts.poppins(fontSize: 13, color: AppColors.success, fontWeight: FontWeight.w500)),
                          ],
                        ),
                      ),
                    ],
                  )
                : Row(
                    children: [
                      if (_isGettingHighPrecisionLocation) ...[
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.primary),
                        ),
                        const SizedBox(width: 12),
                        Text('Getting high-precision location...', style: GoogleFonts.poppins(color: AppColors.textSecondary)),
                      ] else ...[
                        Icon(Icons.location_disabled_rounded, color: AppColors.textSecondary),
                        const SizedBox(width: 12),
                        Text('Location not available', style: GoogleFonts.poppins(color: AppColors.textSecondary)),
                      ],
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

          // Soil Data
          _sectionCard(
            title: 'Soil Properties',
            icon: Icons.science_rounded,
            iconBg: const Color(0xFF2C7A7B),
            child: Column(
              children: [
                _buildValidatedTextField(
                  controller: _soilPhController,
                  labelText: 'Soil pH',
                  hintText: 'e.g. 6.5',
                  fieldKey: 'soil_ph',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _soilNitrogenController,
                  labelText: 'Nitrogen (mg/kg)',
                  hintText: 'e.g. 45.0',
                  fieldKey: 'soil_nitrogen',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _soilOrganicCarbonController,
                  labelText: 'Organic Carbon (%)',
                  hintText: 'e.g. 1.2',
                  fieldKey: 'soil_organic_carbon',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _soilSandController,
                  labelText: 'Sand Content (%)',
                  hintText: 'e.g. 40.0',
                  fieldKey: 'soil_sand',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _soilSiltController,
                  labelText: 'Silt Content (%)',
                  hintText: 'e.g. 35.0',
                  fieldKey: 'soil_silt',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _soilClayController,
                  labelText: 'Clay Content (%)',
                  hintText: 'e.g. 25.0',
                  fieldKey: 'soil_clay',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _soilCecController,
                  labelText: 'Cation Exchange Capacity (meq/100g)',
                  hintText: 'e.g. 15.0 (optional)',
                  fieldKey: 'soil_cec',
                  keyboardType: TextInputType.number,
                ),
              ],
            ),
          ),
          const SizedBox(height: 18),

          // Synthetic Soil Nutrients (Custom Input)
          _sectionCard(
            title: 'Advanced Soil Nutrients',
            icon: Icons.biotech_rounded,
            iconBg: const Color(0xFF7B2CBF),
            child: Column(
              children: [
                _buildValidatedTextField(
                  controller: _ammoniaController,
                  labelText: 'Ammonia (NH₄⁺ mg/kg)',
                  hintText: 'e.g. 2.5',
                  fieldKey: 'ammonia',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _nitrateController,
                  labelText: 'Nitrate (NO₃⁻ mg/kg)',
                  hintText: 'e.g. 3.2',
                  fieldKey: 'nitrate',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _zincController,
                  labelText: 'Zinc (Zn mg/kg)',
                  hintText: 'e.g. 1.2',
                  fieldKey: 'zinc',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _ironController,
                  labelText: 'Iron (Fe mg/kg)',
                  hintText: 'e.g. 18.5',
                  fieldKey: 'iron',
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 14),
                _buildValidatedTextField(
                  controller: _manganeseController,
                  labelText: 'Manganese (Mn mg/kg)',
                  hintText: 'e.g. 12.3',
                  fieldKey: 'manganese',
                  keyboardType: TextInputType.number,
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
            child: _buildValidatedTextField(
              controller: _farmSizeController,
              labelText: 'Area (hectares)',
              hintText: 'e.g. 3.0',
              fieldKey: 'farm_size',
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(height: 28),

          // Validation Error Summary
          if (_validationErrors.isNotEmpty && _tabController.index == 1)
            _buildValidationErrorSummary(),
          
          if (_validationErrors.isNotEmpty && _tabController.index == 1)
            const SizedBox(height: 18),

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
    );
  }

  @override
  void dispose() {
    _tabController.dispose();
    _phosphorusController.dispose();
    _potassiumController.dispose();
    _farmSizeController.dispose();
    _apiUrlController.dispose();
    _soilPhController.dispose();
    _soilNitrogenController.dispose();
    _soilOrganicCarbonController.dispose();
    _soilSandController.dispose();
    _soilSiltController.dispose();
    _soilClayController.dispose();
    _soilCecController.dispose();
    
    // Dispose synthetic nutrient controllers
    _ammoniaController.dispose();
    _nitrateController.dispose();
    _zincController.dispose();
    _ironController.dispose();
    _manganeseController.dispose();
    
    _tabController.dispose();
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

class PredictionResultDialog extends StatefulWidget {
  final Map<String, dynamic> result;
  final double yieldKg;

  const PredictionResultDialog({
    super.key,
    required this.result,
    required this.yieldKg,
  });

  @override
  State<PredictionResultDialog> createState() => _PredictionResultDialogState();
}

class _PredictionResultDialogState extends State<PredictionResultDialog> with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  Widget _resultRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            flex: 2,
            child: Text(label, style: GoogleFonts.poppins(color: AppColors.textSecondary, fontSize: 13)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              value,
              style: GoogleFonts.poppins(fontWeight: FontWeight.w500, color: AppColors.textPrimary, fontSize: 13),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPredictionTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Yield Display
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: [AppColors.success.withOpacity(0.12), AppColors.success.withOpacity(0.05)], begin: Alignment.topLeft, end: Alignment.bottomRight),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.success.withOpacity(0.2)),
            ),
            child: Column(
              children: [
                Text('Expected Yield', style: GoogleFonts.poppins(fontSize: 16, color: AppColors.textSecondary, fontWeight: FontWeight.w500)),
                const SizedBox(height: 8),
                Text(
                  '${widget.yieldKg.toStringAsFixed(2)} kg',
                  style: GoogleFonts.poppins(fontSize: 32, fontWeight: FontWeight.w700, color: AppColors.success),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Location Display
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.grey.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.location_on_rounded, color: AppColors.primary, size: 18),
                    const SizedBox(width: 8),
                    Text('Location', style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                  ],
                ),
                const SizedBox(height: 8),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: Colors.grey.shade300),
                  ),
                  child: Text(
                    widget.result['api_data']['location']['name'].toString(),
                    style: GoogleFonts.poppins(fontSize: 13, color: AppColors.textPrimary),
                    maxLines: 3,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (widget.result['api_data']?['location']?['city'] != null) ...[
                  const SizedBox(height: 6),
                  _resultRow('City', widget.result['api_data']['location']['city'].toString()),
                ],
                if (widget.result['api_data']?['location']?['state'] != null)
                  _resultRow('State', widget.result['api_data']['location']['state'].toString()),
                if (widget.result['api_data']?['location']?['country'] != null)
                  _resultRow('Country', widget.result['api_data']['location']['country'].toString()),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Basic Info
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.grey.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Crop Information', style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                const SizedBox(height: 12),
                _resultRow('Crop Type', (widget.result['input_features']['crop_type'] as String).toString()),
                _resultRow('Region', (widget.result['api_data']['region'] as String).toString()),
                _resultRow('Farm Size', '${widget.result['input_features']['farm_size_ha']} ha'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailedDataTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Weather Data
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.blue.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.blue.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.wb_sunny_rounded, color: Colors.blue.shade700, size: 18),
                    const SizedBox(width: 8),
                    Text('Weather Data', style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.blue.shade700)),
                  ],
                ),
                const SizedBox(height: 12),
                _resultRow('Temperature', '${widget.result['api_data']['weather']['avg_temp']}°C'),
                _resultRow('Rainfall', '${widget.result['api_data']['weather']['rainfall']} mm'),
                _resultRow('Humidity', '${widget.result['api_data']['weather']['humidity_percent']}%'),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Soil Data
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.green.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.green.shade200),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.eco_rounded, color: Colors.green.shade700, size: 18),
                    const SizedBox(width: 8),
                    Text('Soil Properties', style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.green.shade700)),
                  ],
                ),
                const SizedBox(height: 12),
                _resultRow('Soil pH', widget.result['api_data']['soil']['phh2o'].toString()),
                _resultRow('Organic Carbon', '${widget.result['api_data']['soil']['soc']}%'),
                _resultRow('Total Nitrogen', '${widget.result['api_data']['soil']['nitrogen']}%'),
                _resultRow('CEC', '${widget.result['api_data']['soil']['cec']} cmol/kg'),
                _resultRow('Clay Content', '${widget.result['api_data']['soil']['clay']}%'),
                _resultRow('Sand Content', '${widget.result['api_data']['soil']['sand']}%'),
                _resultRow('Silt Content', '${widget.result['api_data']['soil']['silt']}%'),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Advanced Nutrients
          if (widget.result['api_data']?['synthetic_nutrients'] != null) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.purple.shade50,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.purple.shade200),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.biotech_rounded, color: Colors.purple.shade700, size: 18),
                      const SizedBox(width: 8),
                      Text('Advanced Nutrients', style: GoogleFonts.poppins(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.purple.shade700)),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _resultRow('Ammonia (NH₄⁺)', '${widget.result['api_data']['synthetic_nutrients']['ammonia_mg_kg']} mg/kg'),
                  _resultRow('Nitrate (NO₃⁻)', '${widget.result['api_data']['synthetic_nutrients']['nitrate_mg_kg']} mg/kg'),
                  _resultRow('Iron (Fe)', '${widget.result['api_data']['synthetic_nutrients']['iron_mg_kg']} mg/kg'),
                  _resultRow('Manganese (Mn)', '${widget.result['api_data']['synthetic_nutrients']['manganese_mg_kg']} mg/kg'),
                  _resultRow('Zinc (Zn)', '${widget.result['api_data']['synthetic_nutrients']['zinc_mg_kg']} mg/kg'),
                ],
              ),
            ),
          ],

          const SizedBox(height: 24),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () => Navigator.of(context).pop(),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
              child: Text('Close', style: GoogleFonts.poppins(fontSize: 16, fontWeight: FontWeight.w600)),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        color: Colors.white,
      ),
      child: Column(
        children: [
          // Header
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: LinearGradient(colors: [AppColors.primaryDark, AppColors.primary], begin: Alignment.topLeft, end: Alignment.bottomRight),
              borderRadius: const BorderRadius.only(topLeft: Radius.circular(24), topRight: Radius.circular(24)),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(12)),
                  child: Icon(Icons.eco_rounded, color: Colors.white, size: 22),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text('Prediction Results', style: GoogleFonts.poppins(fontSize: 20, fontWeight: FontWeight.w600, color: Colors.white)),
                ),
                IconButton(
                  onPressed: () => Navigator.of(context).pop(),
                  icon: Icon(Icons.close_rounded, color: Colors.white),
                ),
              ],
            ),
          ),

          // Tabs
          Container(
            color: Colors.grey.shade50,
            child: TabBar(
              controller: _tabController,
              indicatorColor: AppColors.primary,
              labelColor: AppColors.primary,
              unselectedLabelColor: AppColors.textSecondary,
              labelStyle: GoogleFonts.poppins(fontWeight: FontWeight.w600),
              tabs: const [
                Tab(text: 'Prediction'),
                Tab(text: 'Detailed Data'),
              ],
            ),
          ),

          // Tab Content
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildPredictionTab(),
                _buildDetailedDataTab(),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

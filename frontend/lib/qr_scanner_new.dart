import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_fonts/google_fonts.dart';

class QRScannerScreen extends StatefulWidget {
  final Function(String) onScanned;

  const QRScannerScreen({Key? key, required this.onScanned}) : super(key: key);

  @override
  State<QRScannerScreen> createState() => _QRScannerScreenState();
}

class _QRScannerScreenState extends State<QRScannerScreen> {
  final MobileScannerController controller = MobileScannerController();
  bool isScanning = true;

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  void _onDetect(BarcodeCapture capture) {
    final List<Barcode> barcodes = capture.barcodes;
    for (final barcode in barcodes) {
      if (barcode.rawValue != null && isScanning) {
        setState(() {
          isScanning = false;
        });
        widget.onScanned(barcode.rawValue!);
        Navigator.of(context).pop();
        break;
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('Scan QR Code', style: GoogleFonts.poppins()),
        backgroundColor: Colors.black,
        iconTheme: const IconThemeData(color: Colors.white),
        actions: [
          IconButton(
            icon: ValueListenableBuilder(
              valueListenable: controller.torchState,
              builder: (context, state, child) {
                switch (state) {
                  case TorchState.off:
                    return const Icon(Icons.flash_off, color: Colors.grey);
                  case TorchState.on:
                    return const Icon(Icons.flash_on, color: Colors.yellow);
                }
              },
            ),
            onPressed: () => controller.toggleTorch(),
          ),
        ],
      ),
      body: Stack(
        children: [
          MobileScanner(
            controller: controller,
            onDetect: _onDetect,
          ),
          CustomPaint(
            size: Size.infinite,
            painter: ScannerOverlay(),
          ),
          Positioned(
            bottom: 100,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Position QR code within the frame',
                  style: GoogleFonts.poppins(
                    color: Colors.white,
                    fontSize: 14,
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

class ScannerOverlay extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0xFF2D6A4F)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final path = Path();
    final scanWindow = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2),
      width: 250,
      height: 250,
    );

    // Draw corners
    final cornerLength = 30.0;
    
    // Top-left corner
    path.moveTo(scanWindow.left, scanWindow.top + cornerLength);
    path.lineTo(scanWindow.left, scanWindow.top);
    path.lineTo(scanWindow.left + cornerLength, scanWindow.top);
    
    // Top-right corner
    path.moveTo(scanWindow.right - cornerLength, scanWindow.top);
    path.lineTo(scanWindow.right, scanWindow.top);
    path.lineTo(scanWindow.right, scanWindow.top + cornerLength);
    
    // Bottom-right corner
    path.moveTo(scanWindow.right, scanWindow.bottom - cornerLength);
    path.lineTo(scanWindow.right, scanWindow.bottom);
    path.lineTo(scanWindow.right - cornerLength, scanWindow.bottom);
    
    // Bottom-left corner
    path.moveTo(scanWindow.left + cornerLength, scanWindow.bottom);
    path.lineTo(scanWindow.left, scanWindow.bottom);
    path.lineTo(scanWindow.left, scanWindow.bottom - cornerLength);

    canvas.drawPath(path, paint);

    // Draw semi-transparent overlay
    final overlayPaint = Paint()..color = Colors.black54;
    final overlayPath = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRect(scanWindow)
      ..fillType = PathFillType.evenOdd;
    
    canvas.drawPath(overlayPath, overlayPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

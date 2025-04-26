import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import '../../const/colors.dart';
import '../uploadingPhoto/updating_screan.dart';

class UploadingScrean extends StatefulWidget {
  const UploadingScrean({super.key});

  @override
  State<UploadingScrean> createState() => _UploadingScreanState();
}

class _UploadingScreanState extends State<UploadingScrean> {
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;
  String _serverUrl = 'http://192.168.1.5:5000';
  String _trainingMessage = '';
  TextEditingController _nameController = TextEditingController();

  Future<void> _pickAndSendImage(ImageSource source, {bool isTraining = false}) async {
    try {
      setState(() {
        _isLoading = true;
        _trainingMessage = '';
      });

      final XFile? image = await _picker.pickImage(source: source);
      if (image == null || !mounted) return;

      var request = http.MultipartRequest(
        'POST',
        Uri.parse(isTraining ? '$_serverUrl/train' : '$_serverUrl/upload'),
      );

      if (isTraining) {
        request.fields['name'] = _nameController.text;
      }
      request.files.add(
        await http.MultipartFile.fromPath('image', image.path),
      );

      final response = await request.send();
      final status = response.statusCode;

      if (status == 200) {
        if (isTraining) {
          // الرد هيكون JSON
          final responseData = await response.stream.bytesToString();
          final jsonResponse = json.decode(responseData);
          _showMessage('تم تدريب النموذج على ${_nameController.text}', isError: false);
          _nameController.clear();
        } else {
          // الرد صورة مباشرة (binary)
          final bytes = await response.stream.toBytes();
          final tempDir = await getTemporaryDirectory();
          final processedImage = File('${tempDir.path}/processed_${DateTime.now().millisecondsSinceEpoch}.jpg');
          await processedImage.writeAsBytes(bytes);

          if (!mounted) return;
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => UpdatingScrean(imageFile: processedImage),
            ),
          );
        }
      } else {
        final error = await response.stream.bytesToString();
        _showMessage('خطأ: ${json.decode(error)['error'] ?? 'حدث خطأ غير معروف'}', isError: true);
      }
    } catch (e) {
      _showMessage('خطأ: ${e.toString()}', isError: true);
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }


  Future<void> _showNameDialog() async {
    return showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('إدخال اسم الشخص'),
          content: TextField(
            controller: _nameController,
            decoration: const InputDecoration(hintText: 'ادخل اسم الشخص'),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('إلغاء'),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                _pickAndSendImage(ImageSource.gallery, isTraining: true);
              },
              child: const Text('موافق'),
            ),
          ],
        );
      },
    );
  }

  void _showMessage(String message, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 3),
        backgroundColor: isError ? AppColors.error : AppColors.success,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: const Text('نظام التعرف على الوجوه', style: TextStyle(color: AppColors.darkText)),
        backgroundColor: AppColors.primary,
        iconTheme: const IconThemeData(color: AppColors.darkText),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 600),
            child: SingleChildScrollView(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (_isLoading)
                    Column(
                      children: [
                        const CircularProgressIndicator(valueColor: AlwaysStoppedAnimation<Color>(AppColors.primary)),
                        const SizedBox(height: 20),
                        Text(
                          _trainingMessage,
                          textAlign: TextAlign.center,
                          style: const TextStyle(fontSize: 16, color: AppColors.darkText),
                        ),
                      ],
                    )
                  else
                    Column(
                      children: [
                        const SizedBox(height: 30),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: AppColors.darkText,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 2,
                            ),
                            onPressed: () => _pickAndSendImage(ImageSource.gallery),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.photo_library, color: AppColors.iconBackground),
                                SizedBox(width: 10),
                                Text('اختر صورة من المعرض', style: TextStyle(fontSize: 16)),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 15),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: AppColors.darkText,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                              elevation: 2,
                            ),
                            onPressed: () => _pickAndSendImage(ImageSource.camera),
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.camera_alt, color: AppColors.iconBackground),
                                SizedBox(width: 10),
                                Text('التقاط صورة جديدة', style: TextStyle(fontSize: 16)),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 30),
                        SizedBox(
                          width: double.infinity,
                          child: OutlinedButton(
                            style: OutlinedButton.styleFrom(
                              foregroundColor: AppColors.primaryDark,
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              side: const BorderSide(color: AppColors.primary),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                              ),
                            ),
                            onPressed: _showNameDialog,
                            child: const Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.person_add, color: AppColors.lightText),
                                SizedBox(width: 10),
                                Text('إضافة شخص جديد', style: TextStyle(fontSize: 16)),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(height: 30),
                        Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: AppColors.iconBackground,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Column(
                            children: [
                              Text(
                                'تعليمات',
                                style: TextStyle(
                                    fontWeight: FontWeight.bold,
                                    color: AppColors.primaryDark,
                                    fontSize: 16),
                              ),
                              SizedBox(height: 10),
                              Text(
                                '• اختر صورة للتعرف على الوجه\n'
                                    '• لإضافة شخص جديد، اضغط "إضافة شخص جديد" واختر صورة\n'
                                    '• يمكنك إضافة عدة صور لكل شخص لتحسين الدقة',
                                textAlign: TextAlign.center,
                                style: TextStyle(color: AppColors.lightText),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
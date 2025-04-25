import 'package:flutter/material.dart';
import 'package:image_project/ui/showingResult/uploading_screan.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
     home: UploadingScrean(),
    );
  }
}
#include "pch.h"
//#include <vector>
//
//using namespace System;
//
//// int main(array<System::String ^> ^args)
//// {
////    return 0;
//// }
//
//#include "Form1.h"
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//using namespace cv;
//using namespace std;
//
//using namespace System::Windows::Forms;
//
//[STAThread]
//// BGR'den Gri Tona D�n���m Fonksiyonu
//void convertBGRToGray(const Mat& input, Mat& output) {
//    // Giri� resminin boyutlar�nda ve tek kanall� (gri ton) bir ��kt� matrisi olu�tur
//    output = Mat(input.rows, input.cols, CV_8UC1);
//
//    for (int i = 0; i < input.rows; i++) {
//        for (int j = 0; j < input.cols; j++) {
//            // Her bir piksel i�in BGR de�erlerini al
//            Vec3b pixel = input.at<Vec3b>(i, j);
//
//            // BGR de�erlerini kullanarak gri ton de�eri hesapla
//            int gray = static_cast<int>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
//
//            // Hesaplanan gri ton de�erini ��kt� matrisine ata
//            output.at<uchar>(i, j) = gray;
//        }
//    }
//}
//float* Conv2D(float* inp, int w, int h, float* kernel, int k, int l) {
//    // ��kt� verilerini saklayacak dinamik bir dizi olu�tur
//    float* output = new float[w * h];
//
//    // Kernel'in yar�s�n� hesapla
//    int kHalf = k / 2;
//    int lHalf = l / 2;
//
//    // Giri� g�r�nt�s� �zerinde dola�
//    for (int y = 0; y < h; ++y) {
//        for (int x = 0; x < w; ++x) {
//            float sum = 0.0;
//
//            // Kernel �zerinde dola�
//            for (int m = -kHalf; m <= kHalf; ++m) {
//                for (int n = -lHalf; n <= lHalf; ++n) {
//                    // Giri� g�r�nt�s�ndeki mevcut konum
//                    int ix = x + m;
//                    int iy = y + n;
//
//                    // Kenar kontrol�
//                    if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
//                        // Kernel ve giri�in ilgili elemanlar�n� �arp ve topla
//                        sum += inp[iy * w + ix] * kernel[(m + kHalf) * k + (n + lHalf)];
//                    }
//                }
//            }
//
//            // Hesaplanan toplam� ��kt� dizisine yaz
//            output[y * w + x] = sum;
//        }
//    }
//
//    return output;
//
//
//
//}
//// Konvol�syon i�lemi i�in fonksiyon prototipi
//float* Conv2D(float* inp, int w, int h, float* kernel, int k) {
//    int pad = k / 2;
//    int outputW = w - k + 1 + 2 * pad;
//    int outputH = h - k + 1 + 2 * pad;
//    float* output = new float[outputW + outputH]();
//
//    // S�f�r dolgusu (zero-padding)
//    for (int y = 0; y < outputH; ++y) {
//        for (int x = 0; x < outputW; ++x) {
//            float sum = 0.0;
//            for (int ky = 0; ky < k; ++ky) {
//                for (int kx = 0; kx < k; ++kx) {
//                    int iy = y + ky - pad;
//                    int ix = x + kx - pad;
//                    if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
//                        sum += inp[iy * w + ix] * kernel[ky * k + kx];
//                    }
//                }
//            }
//            output[y * outputW + x] = sum;
//        }
//    }
//    //// hocan�n yazd��� kod
//
//
//
//    return output;
//}
//
//int main()
//{
//    Application::EnableVisualStyles();
//    Application::SetCompatibleTextRenderingDefault(false);
//    Application::Run(gcnew CppCLRWinFormsProject::Form1());
//
//
//
//
//    Mat image = imread("resim.jpg", IMREAD_COLOR);
//
//    if (!image.data) // G�r�nt� y�klenemediyse kontrol et
//    {
//        cout << "G�r�nt� dosyas� y�klenemedi!" << endl;
//    }
//
//    // G�r�nt�y� pencerede g�ster
//    namedWindow("Test G�r�nt�s�", WINDOW_AUTOSIZE);
//    imshow("Test G�r�nt�s�", image);
//    // Gri tonlu resmi tutacak matrisi tan�mla
//    Mat gray;
//
//    // Kendi yazd���m�z fonksiyonu kullanarak d�n���m� yap
//    convertBGRToGray(image, gray);
//    // Resmi float tipine d�n��t�r
//    Mat imageFloat;
//    image.convertTo(imageFloat, CV_32F, 1.0 / 255); // Normalize edilmi� float'a d�n��t�r
//
//    // Giri� dizisini olu�tur
//    float* input = reinterpret_cast<float*>(imageFloat.data);
//
//    ////Basit bir ortalama alma (blurring) kernel tan�mla
//    int kernelSize = 3;
//    float kernel[] = {
//        1.0f / 16, 2.0f / 16, 1.0f / 16,
//        2.0f / 16, 4.0f / 16, 2.0f / 16,
//        1.0f / 16, 2.0f / 16, 1.0f / 16
//    };
//
//    // Conv2D fonksiyonunu �a��r
//    float* output = Conv2D(input, image.cols, image.rows, kernel, kernelSize, kernelSize);
//
//    // ��kt�y� Mat nesnesine d�n��t�r ve 0-255 aras�na �l�eklendir
//    Mat result(image.rows, image.cols, CV_32F, output);
//    result.convertTo(result, CV_8U, 255); // Geri d�n��t�rme�ve��l�ek
//    ///*�rnek kullan�m*/
//
//    imshow("Gri Tonlu Resim", gray);
//    imshow("bulan�k resim", result);
//
//    waitKey(0); // Kullan�c� bir tu�a basana kadar bekler
//    delete[] output;
//
//
//    return 0;
//}
//
//

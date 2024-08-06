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
//// BGR'den Gri Tona Dönüþüm Fonksiyonu
//void convertBGRToGray(const Mat& input, Mat& output) {
//    // Giriþ resminin boyutlarýnda ve tek kanallý (gri ton) bir çýktý matrisi oluþtur
//    output = Mat(input.rows, input.cols, CV_8UC1);
//
//    for (int i = 0; i < input.rows; i++) {
//        for (int j = 0; j < input.cols; j++) {
//            // Her bir piksel için BGR deðerlerini al
//            Vec3b pixel = input.at<Vec3b>(i, j);
//
//            // BGR deðerlerini kullanarak gri ton deðeri hesapla
//            int gray = static_cast<int>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
//
//            // Hesaplanan gri ton deðerini çýktý matrisine ata
//            output.at<uchar>(i, j) = gray;
//        }
//    }
//}
//float* Conv2D(float* inp, int w, int h, float* kernel, int k, int l) {
//    // Çýktý verilerini saklayacak dinamik bir dizi oluþtur
//    float* output = new float[w * h];
//
//    // Kernel'in yarýsýný hesapla
//    int kHalf = k / 2;
//    int lHalf = l / 2;
//
//    // Giriþ görüntüsü üzerinde dolaþ
//    for (int y = 0; y < h; ++y) {
//        for (int x = 0; x < w; ++x) {
//            float sum = 0.0;
//
//            // Kernel üzerinde dolaþ
//            for (int m = -kHalf; m <= kHalf; ++m) {
//                for (int n = -lHalf; n <= lHalf; ++n) {
//                    // Giriþ görüntüsündeki mevcut konum
//                    int ix = x + m;
//                    int iy = y + n;
//
//                    // Kenar kontrolü
//                    if (ix >= 0 && ix < w && iy >= 0 && iy < h) {
//                        // Kernel ve giriþin ilgili elemanlarýný çarp ve topla
//                        sum += inp[iy * w + ix] * kernel[(m + kHalf) * k + (n + lHalf)];
//                    }
//                }
//            }
//
//            // Hesaplanan toplamý çýktý dizisine yaz
//            output[y * w + x] = sum;
//        }
//    }
//
//    return output;
//
//
//
//}
//// Konvolüsyon iþlemi için fonksiyon prototipi
//float* Conv2D(float* inp, int w, int h, float* kernel, int k) {
//    int pad = k / 2;
//    int outputW = w - k + 1 + 2 * pad;
//    int outputH = h - k + 1 + 2 * pad;
//    float* output = new float[outputW + outputH]();
//
//    // Sýfýr dolgusu (zero-padding)
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
//    //// hocanýn yazdýðý kod
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
//    if (!image.data) // Görüntü yüklenemediyse kontrol et
//    {
//        cout << "Görüntü dosyasý yüklenemedi!" << endl;
//    }
//
//    // Görüntüyü pencerede göster
//    namedWindow("Test Görüntüsü", WINDOW_AUTOSIZE);
//    imshow("Test Görüntüsü", image);
//    // Gri tonlu resmi tutacak matrisi tanýmla
//    Mat gray;
//
//    // Kendi yazdýðýmýz fonksiyonu kullanarak dönüþümü yap
//    convertBGRToGray(image, gray);
//    // Resmi float tipine dönüþtür
//    Mat imageFloat;
//    image.convertTo(imageFloat, CV_32F, 1.0 / 255); // Normalize edilmiþ float'a dönüþtür
//
//    // Giriþ dizisini oluþtur
//    float* input = reinterpret_cast<float*>(imageFloat.data);
//
//    ////Basit bir ortalama alma (blurring) kernel tanýmla
//    int kernelSize = 3;
//    float kernel[] = {
//        1.0f / 16, 2.0f / 16, 1.0f / 16,
//        2.0f / 16, 4.0f / 16, 2.0f / 16,
//        1.0f / 16, 2.0f / 16, 1.0f / 16
//    };
//
//    // Conv2D fonksiyonunu çaðýr
//    float* output = Conv2D(input, image.cols, image.rows, kernel, kernelSize, kernelSize);
//
//    // Çýktýyý Mat nesnesine dönüþtür ve 0-255 arasýna ölçeklendir
//    Mat result(image.rows, image.cols, CV_32F, output);
//    result.convertTo(result, CV_8U, 255); // Geri dönüþtürme ve ölçek
//    ///*Örnek kullaným*/
//
//    imshow("Gri Tonlu Resim", gray);
//    imshow("bulanýk resim", result);
//
//    waitKey(0); // Kullanýcý bir tuþa basana kadar bekler
//    delete[] output;
//
//
//    return 0;
//}
//
//

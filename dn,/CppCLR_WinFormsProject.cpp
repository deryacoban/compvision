#include "pch.h"
#include <vector>

#include <fstream>

using namespace System;

// int main(array<System::String ^> ^args)
// {
//    return 0;
// }

#include "Form1.h"
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

// BGR'den Gri Tona Dönüþüm Fonksiyonu
void convertBGRToGray(const Mat& input, Mat& output) {
    output = Mat(input.rows, input.cols, CV_8UC1);
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            Vec3b pixel = input.at<Vec3b>(i, j);
            int gray = static_cast<int>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            output.at<uchar>(i, j) = gray;
        }
    }
}

// Düzeltme yapýlmýþ Konvolüsyon Fonksiyonu
void Conv2D(const Mat& input, Mat& output, const vector<vector<float>>& kernel) {
    int kernelSize = kernel.size();
    int pad = kernelSize / 2;
    output = Mat(input.rows, input.cols, input.type());

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            float sum = 0.0;

            for (int m = -pad; m <= pad; m++) {
                for (int n = -pad; n <= pad; n++) {
                    int x = j + n;
                    int y = i + m;

                    // Kenar kontrolü
                    if (x >= 0 && x < input.cols && y >= 0 && y < input.rows) {
                        sum += input.at<uchar>(y, x) * kernel[pad + m][pad + n];
                    }
                }
            }

            output.at<uchar>(i, j) = min(max(int(sum), 0), 255);
        }
    }
}




// K-Means Clustering Fonksiyonu
//void applyKMeansClustering(const Mat& input, Mat& output, int k) {
//    Mat data;
//    input.convertTo(data, CV_32F);
//    data = data.reshape(1, data.total());
//
//    // K-means clustering
//    Mat labels, centers;
//    kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
//
//    // Cluster merkezlerini kullanarak output görüntüsünü oluþtur
//    centers = centers.reshape(1, centers.rows);
//    output = Mat(input.size(), input.type());
//
//    for (int i = 0; i < input.rows; i++) {
//        for (int j = 0; j < input.cols; j++) {
//            int cluster_id = labels.at<int>(i * input.cols + j);
//            output.at<uchar>(i, j) = centers.at<float>(cluster_id, 0);
//        }
//    }
//}
//unsigned char* inp(int path, int& width, int& height);
//int* hist = histogram(inp, width, height);
// 
// 
// 
// 
// // K-Means Clustering Fonksiyonu
void applyKMeansClustering(const Mat& input, Mat& output, int k) {
    Mat data;
    input.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    // K-means clustering
    Mat labels, centers;
    kmeans(data, k, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // Cluster merkezlerini kullanarak output görüntüsünü oluþtur
    centers = centers.reshape(1, centers.rows);
    output = Mat(input.size(), input.type());

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            int cluster_id = labels.at<int>(i * input.cols + j);
            output.at<uchar>(i, j) = centers.at<float>(cluster_id, 0);
        }
    }
}

// Mahalanobis Mesafesi Hesaplama Fonksiyonu
double mahalanobisDistance(const Vec3f& vec, const Vec3f& mean, const Mat& invCovar) {
    Mat diff = (Mat_<float>(3, 1) << vec[0] - mean[0], vec[1] - mean[1], vec[2] - mean[2]);
    Mat mahal = diff.t() * invCovar * diff;
    return sqrt(mahal.at<float>(0, 0));
}

// Fonksiyon: Görüntüyü dosyadan okur ve bir unsigned char dizisi olarak döndürür.
unsigned char* loadImage(const string& path, int& width, int& height) {
    ifstream file(path, ios::binary | ios::ate);

    // Dosya boyutunu bul.
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    // Görüntüyü hafýzada saklamak için bir dizi ayýr.
    unsigned char* buffer = new unsigned char[size];
    if (file.read((char*)buffer, size)) {
        //  geniþlik ve yüksekliðe sabit deðerler atandý.
        width = 256; // Örnek geniþlik
        height = 256; // Örnek yükseklik
        return buffer;
    }
    else {
        // Okuma baþarýsýz olursa
        delete[] buffer;
        return nullptr;
    }
}

//  Histogramý hesapla
int* computeHistogram(unsigned char* image, int width, int height) {
    int* histogram = new int[256]();

    for (int i = 0; i < width * height; i++) {
        histogram[image[i]]++;
    }

    return histogram;
}

// Histogramý ekrana yazdýr
void printHistogram(int* histogram) {
    for (int i = 0; i < 256; i++) {
        cout << i << ": " << histogram[i] << endl;
    }
}

int main() {
    string path = "resim.jpg";
    int width, height;
    unsigned char* imagehist = loadImage(path, width, height);
    if (imagehist != nullptr) {
        int* histogram = computeHistogram(imagehist, width, height);
        printHistogram(histogram);
        // Sonra belleði serbest býrak
        delete[] histogram;

        // imagehist'den cv::Mat oluþturma
        Mat histImage(height, width, CV_8UC1, imagehist);
        imshow("Histogram Image", histImage);
        delete[] imagehist;
    }
    else {
        cout << "Görüntü dosyasý yüklenemedi!" << endl;
    }
    Mat image = imread("resim.jpg", IMREAD_COLOR);


    Mat gray;
    convertBGRToGray(image, gray);

    // Ortalama alma kernel'i tanýmla
    vector<vector<float>> kernel = {
        {1.0 / 16, 2.0 / 16, 1.0 / 16},
        {2.0 / 16, 4.0 / 16, 2.0 / 16},
        {1.0 / 16, 2.0 / 16, 1.0 / 16}
    };
    /* Mat image = imread("resim.jpg", IMREAD_GRAYSCALE);*/
    if (image.empty()) {
        cout << "Görüntü dosyasý yüklenemedi!" << endl;
        return -1;
    }

    //Mat clustered;
    //applyKMeansClustering(image, clustered, 3);  // Örneðin, 3 cluster kullanarak


    /*imshow("Clustered Image", clustered);*/
    Mat blurredImage;
    Conv2D(gray, blurredImage, kernel);

    imshow("Original Image", image);
    imshow("Gray Image", gray);
    imshow("Blurred Image", blurredImage);


    // Mahalanobis distance hesaplama örneði
    Mat samples;
    image.convertTo(samples, CV_32F);
    samples = samples.reshape(1, image.total());
    Mat covar, mean;
    calcCovarMatrix(samples, covar, mean, COVAR_NORMAL | COVAR_ROWS);
    Mat invCovar;
    invert(covar, invCovar);

    Vec3f sample(1.0, 2.0, 3.0); // Örnek bir vektör
    Vec3f meanVec(mean.at<float>(0, 0), mean.at<float>(0, 1), mean.at<float>(0, 2)); // Mean vektörü
    double dist = mahalanobisDistance(sample, meanVec, invCovar);
    cout << "Mahalanobis Distance: " << dist << endl;

    waitKey(0);
    return 0;
}
#include <boost/mpl/size.hpp>
#include <string.h>
#include "cvwld.h"
#include "highgui.h"
#include <assert.h>
#include <fstream>
#include<iostream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include<string>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace std;
void wld_feat_save(string dest_path, string path){
    DIR *dp;
    struct dirent *dirp;
    fstream feats;
    string image_path;
    fstream names;
    string file_nms = "Filenms.txt";
    file_nms = path+file_nms;
    cout<<file_nms<<endl;
    names.open(file_nms, ios::out);
    feats.open(dest_path, ios::out);
    dp  = opendir(path.c_str());
    int img_count = 0;  //for sanity check
    while((dirp = readdir(dp))!=NULL){
        if(strcmp(dirp->d_name,".")&&strcmp(dirp->d_name,"..")&&strcmp(dirp->d_name,"Filenms.txt")){
            image_path = path+dirp->d_name;
            names<<image_path<<endl;
            img_count++;
            try{
                IplImage* image = cvLoadImage(image_path.c_str());
                IplImage* gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
                cvCvtColor(image, gray, CV_BGR2GRAY);
                CvMemStorage* storage = cvCreateMemStorage(0);
                CvSeq* keypoints;
                CvSeq* descriptors;
                CvMat* hr = cvExtractWLD(gray, cvWLDParams(8, 6, 20, 100, cvSize(64, 64)));
                for (int i = 0; i < 20 * 6 * 8; i++)
                {
                    feats<<100 - hr->data.fl[i] * 10000<<" ";
                }
                feats<<endl;
            }
            catch(...){
            }
        }
    }    
    cout<<path<<" "<<img_count<<endl;
    names.close();
    feats.close();
}
int main(){
    string path_train("../../race_d/cropped/train/");
    string dest_path_train("../../race_d/wld_feat/cropped/train/");
    string path_valid("../../race_d/cropped/validate/");
    string dest_path_valid("../../race_d/wld_feat/cropped/validate/");
    string path_test("../../race_d/cropped/test/");
    string dest_path_test("../../race_d/wld_feat/cropped/test/");
    string labels[2] = {"indian/","chinese/"};
    string paths[2] = {path_train, path_test};
    
    string path;    //for path to images
    string dest_path;   //for path to feature file saving
    /*
     * Open Indian training set for feature extraction
     */
    path = path_train+labels[0];
    dest_path = dest_path_train+labels[0]+"feats";
    wld_feat_save(dest_path, path);
    /*
     * Chinese training
     */
    path = path_train+labels[1];
    dest_path = dest_path_train+labels[1]+"feats";
    wld_feat_save(dest_path, path);
    /*
     * Validate set features - indian
     */
    path = path_valid+labels[0];
    dest_path = dest_path_valid+labels[0]+"feats";
    wld_feat_save(dest_path,path);
    /*
     * validate set features - chinese
     */
    path = path_valid+labels[1];
    dest_path = dest_path_valid+labels[1]+"feats";
    wld_feat_save(dest_path,path);
    /*
     * Test set features - indian
     */
    path = path_test+labels[0];
    dest_path = dest_path_test+labels[0]+"feats";
    wld_feat_save(dest_path,path);
    /*
     * Test set features - chinese
     */
    path = path_test+labels[1];
    dest_path = dest_path_test+labels[1]+"feats";
    wld_feat_save(dest_path,path);
    return 0;
}

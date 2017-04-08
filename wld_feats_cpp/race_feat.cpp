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
using namespace std;
void wld_feat_save(string dest_path, string path){
    DIR *dp;
    struct dirent *dirp;
    fstream feats;
    string image_path;
    feats.open(dest_path, ios::out);
    dp  = opendir(path.c_str());
    int img_count = 0;  //for sanity check
    while((dirp = readdir(dp))!=NULL){
        if(strcmp(dirp->d_name,".")&&strcmp(dirp->d_name,"..")){
            image_path = path+dirp->d_name;
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
    feats.close();
}
int main(){
    string path_train("../../race_d/train/");
    string dest_path_train("../../race_d/wld_feat/train/");
    string path_test("../../race_d/test/");
    string dest_path_test("../../race_d/wld_feat/test/");
    string labels[2] = {"indian/","chinese/"};
    string paths[2] = {path_train, path_test};
    /*
     * Open Indian training set for feature extraction
     */
    string path;    //for path to images
    string dest_path;   //for path to feature file saving
//     path = path_train+labels[0];
//     dest_path = dest_path_train+labels[0]+"feats";
//     wld_feat_save(dest_path, path);
    /*
     * Chinese training
     */
//     path = path_train+labels[1];
//     dest_path = dest_path_train+labels[1]+"feats";
//     wld_feat_save(dest_path, path);
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

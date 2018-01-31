#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "map_pgm.h"
#include <sys/stat.h>
#include <dirent.h>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>


struct data
{
    int vec[784];
    int label;
    data (const Pgm_map & p)
    {
        label = p.label;
        int i = 0;
        for (int x = 0; x < p.getSizeX(); x++)
        {
            for (int y = 0; y< p.getSizeY(); y++)
            {
                
                vec[i++] = 1 - p.getGridMap2D()[x][y];
                // std::cout << vec[i++] << " ";
            }
        }
        //memset(vec,0,784);
    }
    data (std::string str)
    {
        using namespace std;
        stringstream ss;
        ss << str;
        int i = 0;
        ss >> i ;
        label = i;
        //cout << label << endl;
        int j = 0;
        while(ss >> i)
        {
            vec[j++] = i;
        }
    }
    data & operator = (const data &d)
    {
        label = d.label;
        memcpy(vec, d.vec, 784 * sizeof(int));
        return *this;
    }
    void writeToFile()
    {
        using namespace std;
        ofstream fout;
        fout.open("trained.txt",ios::app);
        fout << label << " ";
        for (int i = 0; i < 784; i++)
        {
            fout << vec[i] << " ";
            // cout << vec[i] << " ";
        }
        fout << "\n";
        fout.close();
    }
};

void getFiles(const char* path, std::vector<std::string>& files){

    using namespace std;
    const string path0 = path;
    DIR* pDir;
    struct dirent* ptr;

    struct stat s;
    lstat(path, &s);

    if(!S_ISDIR(s.st_mode)){
        cout << "not a valid directory: " << path << endl;
        return;
    }

    if(!(pDir = opendir(path))){
        cout << "opendir error: " << path << endl;
        return;
    }
    int i = 0;
    string subFile;
    while((ptr = readdir(pDir)) != 0){
        subFile = ptr -> d_name;
        if(subFile == "." || subFile == "..")
            continue;
        subFile = subFile;
        //cout << ++i << ": " << subFile << endl;
        files.push_back(subFile);
    }
    closedir(pDir);

}


bool dataTrain()
{
    using namespace std;
    vector<string> trainFiles;
    getFiles("mnist_data",trainFiles);
    cout << trainFiles.size() << endl;
    cout << trainFiles[0] << endl;
    for(auto s : trainFiles)
    {
        string filepath = "mnist_data/"; 
        filepath.append(s);   
        Pgm_map p(filepath.c_str());
        p.transformMapToGrid();
        data d(p);
        d.writeToFile();
    }


    // string filepath = "mnist_data/";
    // filepath.append(trainFiles[0]);
    // Pgm_map p(filepath.c_str());
    // p.transformMapToGrid();
    // data d(p);
    // d.writeToFile();

    return true;
};

void loadTrainedData(std::vector<data> &vd)
{
    using namespace std;
    ifstream fin;
    fin.open("trained.txt");
    string str;
    while ( getline(fin,str) )
    {
        vd.push_back(data(str));
    }
    //cout << vd[1].vec[3];

    fin.close();
}

double getDistanceOfVector(const int * v1, const int * v2)
{
    double result = 0;
    for (int i =0 ; i < 784 ; i++)
    {
        result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    // std::cout << result << " ";
    return result;
}

struct result
{
    int label;
    double distance;
    result (int _label, double _distance)
    {
        label = _label;
        distance = _distance;
    }
};

void recgonizeNum(const data &d, const std::vector<data> &vd, int k, std::vector<result> &vr)
{
    using namespace std;
    //vector <result> vr(k);
    for (auto dinv : vd)
    {
        double dis = getDistanceOfVector(d.vec,dinv.vec);
        if (vr.size() < k)
        {
            vr.push_back(result(dinv.label,dis));
        }
        else
        {
            for (result & r : vr)
            {
                if (dis < r.distance)
                {
                    // cout << r.label << " and(panduan) " << r.distance << endl;
                    r.label = dinv.label;
                    r.distance = dis;
                    break;
                }
            }
        }
    }
}

int main(int argc, char * argv[])
{
    using namespace cv;
    using namespace std;
    vector<data> vd;
    vector<result> vr;
    // dataTrain();
    loadTrainedData(vd);

    // for (auto d : vd)
    // {
    //     cout << d.label << endl;
    // }

    Pgm_map p(argv[1]);
    p.transformMapToGrid();
    data testdata(p);
    recgonizeNum(testdata, vd, 10, vr);
    for(result r : vr)
    {
        cout << r.label << endl;
    }
    return 0;
}
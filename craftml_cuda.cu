#include <iostream>
#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <string>
#include "math.h"
#include "limits.h"

#define MIN -99
#define M 104
#define N 1500
#define trainFileName "train_full.csv"
#define testFileName "test_full.csv"
#define features 55
#define K 10
#define trainData(row,col) trainData[col+row*M]
#define testData(row,col) testData[col+row*M]
#define THRESHOLD 0
using namespace std;

vector <vector <double> > train_file;
vector <vector <double> > test_file;

vector <vector <double> > trainFile_Full;
vector <vector <double> > testFile_Full;


int *device_trainFileData, *device_cardinality;
float *infoGainsInitializer;
__device__ float device_infoGainOfData;

dim3 blocks(M);
dim3 threads(N);

struct Node{
	int number_of_children;
	int branch_value;
	int split_attribute;
	int leaf_value;
	struct Node *children[10];
};

typedef struct Node node;

node* create(){
	node* n = new node;
	n->number_of_children = 0;
	n->branch_value = -1;
	n->split_attribute = -1;
	n->leaf_value = -1;
	return n;
}
void chooseRandomFeatures(){
	vector<vector<double> > trainFileRandom( N , vector<double> (M, 0));
	for(int i=0; i<50; i++){
		int guess = rand() % 103; 

		trainFileRandom[i]=train_file[guess];
	}
	train_file=trainFileRandom;	
}
double cosine_distance(double *A, double *B,   int Vector_Length)
{
    double dot = 0.0,  denominator_a = 0.0,  denominator_b = 0.0 ;
    for(  int i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i] ;
        denominator_a += A[i] * A[i] ;
         denominator_b += B[i] * B[i] ;
    }
    return dot / (sqrt( denominator_a) * sqrt( denominator_b)) ;
}
void k_means(){
	double  trainFile1[N][M];
    int minima[features]={INT_MAX};
    int maxima[features]={INT_MIN};
    int cluster[N];
    int t=0;
    for(int i=0; i<N; i++){
        for(int j=0; j<features; j++){
            if( trainFile1[i][j]<minima[j]){
                minima[j]= trainFile1[i][j];
            }
            if( trainFile1[i][j]>maxima[j]){
                maxima[j]= trainFile1[i][j];
            }
        }
    }
    double mean_arr[K][features];
    for(int i=0; i<K; i++){
        for(int j=0; j<features; j++){
            int num = (rand() % (maxima[j] - minima[j] + 1)) + minima[j]; 
            mean_arr[i][j]=num;
        }
    }

    for (int i = 0; i < t; i++) {
        for (int j = 0; j < N; j++) {
            double* dists = new double[K];
            for (int p = 0; p < K; p++) {
                dists[p] = cosine_distance( trainFile1[j], mean_arr[p], M);
            }
            cluster[j] = std::min_element(dists, dists + K) - dists;
            delete[] dists;
        }
        double sum[K][M]={0};
        int count[K]={0};
        
        for (int f = 0; f < N; f++) {
            for (int p = 0; p < M; p++) {
                sum[cluster[f]][p]+= trainFile1[f][p];
            }
            count[cluster[f]]++;
        }
        for (int f = 0; f < K; f++) {
            for (int p = 0; p < M; p++) {
                mean_arr[f][p]=sum[f][p]/count[f];
            }
        }
    }
}

  void read_files(string file_name){
	if(file_name.compare("training")==0){
		string line;
		ifstream ifs(trainFileName);
		while(getline(ifs,line)){
			vector <double> entry;
			stringstream lineStream(line);
			string value;
			while(getline(lineStream,value,',')){
				entry.push_back(stof(value));
			}
			train_file.push_back(entry);
		}
		ifs.close();
	}
	else if(file_name.compare("testing")==0){
		string line1;
		ifstream ifs1(testFileName);
		while(getline(ifs1,line1)){
			vector <double> entry;
			stringstream lineStream1(line1);
			string value;
			while(getline(lineStream1,value,',')){
				entry.push_back(stof(value));
			}
			test_file.push_back(entry);
		}
		ifs1.close();
	}
  }

  __global__ void getInformationGains(int *attr,int *data,int dataSize,float *infoGains,int *trainData,int *cardinality)
{
	if(attr[blockIdx.x]==0 && blockIdx.x!=0 && blockIdx.x!=M-1){
		int threadid,blockid,j;
		threadid=threadIdx.x;
		blockid=blockIdx.x;
		__shared__ int value_attribute[10];
		__shared__ int value_class_attribute[10][10];
		if(threadid<10){
			value_attribute[threadid]=0;
			for(j=0;j<10;j++){
				value_class_attribute[threadid][j]=0;
			}
		}
		__syncthreads();
		int classVal = trainData(data[threadid],M-1);
		int attribute_value = trainData(data[threadid],blockid);
		atomicAdd(&value_attribute[attribute_value],1);
		atomicAdd(&value_class_attribute[attribute_value][classVal],1);
		__syncthreads();
		if(threadid==0){
			int i,j;
			float information_gain,intermediateGain;
			information_gain=0;
			for(i=1;i<=cardinality[blockid];i++){
				intermediateGain=0;
				if(value_attribute[i]==0){
					continue;
				}
				for(j=1;j<=cardinality[M-1];j++){
					if(value_class_attribute[i][j]==0){
						continue;
					}
					intermediateGain+=(float(value_class_attribute[i][j])/(float)value_attribute[i])*(log((float)value_class_attribute[i][j]/(float)value_attribute[i])/log((float)2));
				}
				intermediateGain*=(float(value_attribute[i])/(float)dataSize);
				information_gain-=intermediateGain;
			}
			infoGains[blockid]=information_gain;
		}
	}
}

__global__ void getInfoGainOfData(int *data,int dataSize,int *trainData,int *cardinality)
{
	__shared__ int value_class_count[10];
	int classVal,i,threadid;
	float information_gain;
	threadid=threadIdx.x;
	if(threadid<10){
		value_class_count[threadid]=0;
	}
	__syncthreads();
	classVal=trainData(data[threadIdx.x],M-1);
	atomicAdd(&value_class_count[classVal],1);
	__syncthreads();
	if(threadid==0){
		information_gain=0;
		for(i=1;i<=cardinality[M-1];i++){
			if(value_class_count[i]==0){
				continue;
			}
			information_gain+=((float)value_class_count[i]/(float)dataSize)*(log((float)value_class_count[i]/(float)dataSize)/log((float)2));
		}
		device_infoGainOfData=-1*information_gain;
	}
}

int majority_vote(int *data,int dataSize)
{
	int i,outputClass,ans,maxVal;
	map <int, int> dataCount;
	map <int, int>::iterator iterator;
	for(i=0;i<dataSize;i++){
		outputClass = train_file[data[i]][M-1];
		if(dataCount.find(outputClass)==dataCount.end()){
			dataCount.insert(make_pair(outputClass,1));
		}
		else{
			dataCount[outputClass]++;
		}
	}
	maxVal = MIN;
	for(iterator=dataCount.begin();iterator!=dataCount.end();iterator++){
		if(iterator->second > maxVal){
			ans = iterator->first;
		}
	}
	return ans;
}

void make_decision(int *host_attributes, int *host_data, node *root, int host_datasize)	{
	int flag, host_selectedAttribute, i;
	k_means();
	if(host_datasize<=THRESHOLD){
		return;
	}
	float maxGain;
	flag=1;
	
	for(i=1;i<host_datasize;i++){
		if(train_file[host_data[i]][M-1]!=train_file[host_data[i-1]][M-1]){
			flag=0;
			break;
		}
	}
	if(flag==1){
		root->leaf_value=train_file[host_data[0]][M-1];
		return;
	}
	
	int *device_attr, *device_data;
	float *device_infoGains;
	float host_informationGains[M];
	float host_infoGainOfData;

	cudaMalloc((void**)&device_attr,M*sizeof(int));
	cudaMalloc((void**)&device_data,host_datasize*sizeof(int));
	cudaMalloc(&device_infoGains,M*sizeof(float));
	cudaMemcpy((void*)device_attr,(void*)host_attributes,M*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)device_data,(void*)host_data,host_datasize*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(device_infoGains, infoGainsInitializer, M*sizeof(float),cudaMemcpyHostToDevice);

	getInformationGains<<<blocks,host_datasize>>>(device_attr,device_data,host_datasize,device_infoGains,device_trainFileData,device_cardinality);
	
	cudaMemcpy((void*)host_informationGains,(void*)device_infoGains,M*sizeof(float),cudaMemcpyDeviceToHost);

	cudaFree(device_attr);
	cudaFree(device_infoGains);

	getInfoGainOfData<<<1,host_datasize>>>(device_data,host_datasize,device_trainFileData,device_cardinality);

	cudaMemcpyFromSymbol(&host_infoGainOfData,device_infoGainOfData,sizeof(float),0,cudaMemcpyDeviceToHost);

	cudaFree(device_data);

	maxGain=MIN;
	host_selectedAttribute=-1;
	for(i=1;i<M-1;i++){
		if(host_attributes[i]==0){
			host_informationGains[i]=host_infoGainOfData-host_informationGains[i];
			if(host_informationGains[i]>maxGain){
				maxGain=host_informationGains[i];
				host_selectedAttribute=i;
			}
		}
	}

	root->split_attribute = host_selectedAttribute;
	host_attributes[host_selectedAttribute]=1;

	if(host_selectedAttribute==-1){
		root->leaf_value = majority_vote(host_data, host_datasize);
		return;
	}

	map<int, vector <int> > dividedData;
	map<int, vector <int> >::iterator iterator;
	int attribute_value;

	for(i=0;i<host_datasize;i++){
		attribute_value = train_file[host_data[i]][host_selectedAttribute];
		if(dividedData.find(attribute_value) == dividedData.end()){
			vector <int> x;
			x.push_back(host_data[i]);
			dividedData.insert(make_pair(attribute_value,x));
		}
		else{
			dividedData[attribute_value].push_back(host_data[i]);
		}
	}
	for(i=0,iterator=dividedData.begin(); iterator!=dividedData.end(); iterator++,i++){
		root->number_of_children++;
		node* childNode;
		childNode = create();
		
		root->children[i] = childNode;
		childNode->branch_value = iterator->first;
		

		int new_attr[M];
		for(int z=0;z<M;z++){
			new_attr[z]=host_attributes[z];
		}
		int* host_childData = &(iterator->second[0]);
		make_decision(new_attr, host_childData, childNode, iterator->second.size());
	}
}

__global__ void getCardinality(int *trainData, int *cardinality)	{
	__shared__ int x[10];
	int blockid, threadid,i;
	blockid = blockIdx.x;
	threadid = threadIdx.x;
	if(threadid<10){
		x[threadid]=0;
	}
	__syncthreads();
	if(blockIdx.x!=0){
		x[trainData(threadid, blockid)] = 1;
		__syncthreads();
		for(i=1;i<10;i*=2){
			int index = 2*i*threadid;
			if(index+i<10){
				x[index]+=x[index+i];
			}
			__syncthreads();
		}
		if(threadid==0){
			cardinality[blockid]=x[0];
		}
	}
	__syncthreads();
}

void fillTrainFile(vector <vector <double> > trainFile_Full, int index){
    for(int j=0; j<trainFile_Full.size(); j++){
        vector<double> temp;
        for(int i=0; i<(M-1); i++){
            temp.push_back(trainFile_Full[j][i]);
        }
        temp.push_back(trainFile_Full[j][index]);
        train_file.push_back(temp);
    }
}
void fillTestFile(vector <vector <double> > testFile_Full, int index){
    for(int j=0; j<testFile_Full.size(); j++){
        vector<double> temp;
        for(int i=0; i<(M-1); i++){
            temp.push_back(testFile_Full[j][i]);
        }
        temp.push_back(testFile_Full[j][index]);
        test_file.push_back(temp);
    }
}

void test(node* root, int index)	{
	int i,pos,neg,noResult,attr,attribute_value,j,flag;
	node* temp;
	pos=0;
	neg=0;
	noResult=0;
	// readCSV("testing");
	// read_files("testing");
	fillTestFile(testFile_Full, index);

	for(i=0;i<test_file.size();i++){
		temp=root;
		flag=0;
		while(temp->leaf_value==-1 && temp->split_attribute!=-1){
			attr = temp->split_attribute;
			attribute_value=test_file[i][attr];
			for(j=0;j<temp->number_of_children;j++){
				if(temp->children[j]->branch_value == attribute_value){
					break;
				}
			}
			if(j==temp->number_of_children){
				flag=1;
				break;
			}
			else{
				temp=temp->children[j];
			}
		}
		if(temp->leaf_value == test_file[i][M-1]){
			pos++;
		}
		else{
			neg++;
		}
		if(temp->leaf_value == -1 || flag==1){
			noResult++;
		}
	}
	cout << "Class" << (index - 102) << "  :  ";
	cout << "Accuracy: " << max(pos, neg)/(pos+neg+0.0)*1.0;
	return;
}

void extractFull(string str)	{
	if(str.compare("training")==0){
		ifstream ifs(trainFileName);
		string line;
		
		while(getline(ifs,line)){
			stringstream lineStream(line);
			string cell;
			vector <double> values;
			while(getline(lineStream,cell,',')){
				values.push_back(stof(cell));
			}
			trainFile_Full.push_back(values);
		}
		ifs.close();
	}
	else if(str.compare("testing")==0){
		ifstream ifs1(testFileName);
		string line1;
		
		while(getline(ifs1,line1)){
			stringstream lineStream1(line1);
			string cell1;
			vector <double> values1;
			while(getline(lineStream1,cell1,',')){
				values1.push_back(stof(cell1));
			}
			testFile_Full.push_back(values1);
		}
		ifs1.close();
	}
}

int main()	{
	int i;
	node* root;
	extractFull("training");
	extractFull("testing");

	// readCSV("training");
	// read_files("training");

	for(int index=103; index<117; index++){

		train_file.clear();
		test_file.clear();
		

		fillTrainFile(trainFile_Full, index);

		// chooseRandom();

		int host_trainFileData[N*M+5]={0};
		
		for(i=0;i<N*M;i++){
			host_trainFileData[i] = train_file[i/M][i%M];
		}
		
		int host_data[N], host_attributes[M];

		for(i=0;i<N;i++){
			host_data[i]=i;
		}
		for(i=0;i<M;i++){
			host_attributes[i]=0;
		}

		cudaMalloc((void**)&device_trainFileData, N*M*sizeof(int));
		cudaMemcpy((void*)device_trainFileData,(void*)host_trainFileData, M*N*sizeof(int), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&device_cardinality,M*sizeof(int));
		cudaMemset(device_cardinality, 0, M*sizeof(int));
	
		getCardinality<<<blocks,threads>>>(device_trainFileData, device_cardinality);

		root = create();

		infoGainsInitializer = (float*)malloc( M * sizeof(float));
		for(i=0; i<M; i++){
			infoGainsInitializer[i]=MIN;
		}
		
	
		make_decision(host_attributes, host_data, root, N);


		cudaFree(device_trainFileData);
		cudaFree(device_cardinality);

		test(root, index);

		cout << endl;
	
	}
	return 0;
}
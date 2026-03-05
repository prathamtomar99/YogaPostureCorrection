#include <bits/stdc++.h>
using namespace std;
int main(){
    string ss;
    cin>>ss;
  
    // dream, dreamer, erase and eraser.
    string s = "dream";
    int i=0;
    while(i<s.size()){
        int start = i;
        if(ss[start] == 'd'){
            if(start + 5 >= ss.size() || ss.substr(start,5) != s){
                cout<<"NO";
                return;
            }
            start+=5;
            if(start == ss.size()) break;
            if(start + 4 >= ss.size()){
                cout<<"NO";
                return;
            }
            if(ss[start+1] == 'd') continue;
            else if(ss[start+1] == 'e' && ss[start+2] == 'r') {
                
                continue;
            }
            else{
                cout<<"NO";
                return;
            }
        }
        if()
    }
}
#include <iostream>
#include <libsvm/svm.h>
using namespace std;

int main() {
    struct svm_node *x = new svm_node;
    x->index = 0;
    x->value = 10;
    cout << x->index << x->value << endl;
    return 0;
}
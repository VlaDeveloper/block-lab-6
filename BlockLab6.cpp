#include <cmath>
#include <iostream>
#include <ctime>
#include <omp.h>
#include <cstdlib>
#include <limits>
#include "oneapi/tbb/parallel_for.h"

using namespace std;

int n;

bool is_equal(double x, double y) {
    return fabs(x - y) < 0, 00000001;
}

void FillAArr(double** a, int n) {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            a[i][j] = rand() % 100;
        }
    }
}

void FillXArr(double* x, int n) {
    for (int i = 1; i <= n; i++) {
        x[i] = rand() % 100;
    }
}

void FillBArr(double** a, double* x, int n) {
    for (int i = 1; i <= n; i++) {
        a[i][n + 1] = 0;
        for (int j = 1; j <= n; j++) {
            a[i][n + 1] += a[i][j] * x[j];
        }
    }
}

void Gauss(double** a, double* x, int n) {
    /* Прямой ход*/
    unsigned int start = clock();
    for (int k = 1; k < n; k++) {
        for (int j = k; j < n; j++) {
            double d = a[j][k - 1] / a[k - 1][k - 1];
            for (int i = 0; i <= n; i++) {
                a[j][i] = a[i][i] - d * a[k - 1][i];
            }
        }
    }

    /*Обратный ход*/
    for (int i = n - 1; i >= 0; i--) {
        x[i] = a[i][n] / a[i][i];
        for (int j = n - 1; j > i; j--) {
            x[i] = x[i] - a[i][j] * x[j] / a[i][i];
        }
    }
    cout << "Гаусс без распараллеливания: " << clock() - start << endl;
}

bool CheckAnswers(double* x, double* x2, int n) {
    for (int i = 1; i <= n; i++) {
        if (!is_equal(x[i], x2[i]))
            return false;
    }
    return true;
}

void GaussParallel(double** a, double* x, int n) {
    /* Прямой ход*/
    unsigned int start = clock();
    for (int k = 1; k < n; k++) {
#pragma omp parallel for
        for (int j = k; j < n; j++) {
            double d = a[j][k - 1] / a[k - 1][k - 1];
            for (int i = 0; i <= n; i++) {
                a[j][i] = a[i][i] - d * a[k - 1][i];
            }
        }
    }

    /*Обратный ход*/
    for (int i = n - 1; i >= 0; i--) {
        x[i] = a[i][n] / a[i][i];
        for (int j = n - 1; j > i; j--) {
            x[i] = x[i] - a[i][j] * x[j] / a[i][i];
        }
    }
    cout << "Гаусс c распараллеливанием OMP: " << clock() - start << endl;
}

void GaussParallel2(double** a, double* x, int n) {
    /* Прямой ход*/
    unsigned int start = clock();
    for (int k = 1; k < n; k++) {
        tbb::parallel_for(tbb::blocked_range<int>(k, n),
            [&](tbb::blocked_range<int> r)
            {
                for (int j = r.begin(); j < r.end(); j++)
                {
                    double d = a[j][k - 1] / a[k - 1][k - 1];
                    for (int i = 0; i <= n; i++) {
                        a[j][i] = a[i][i] - d * a[k - 1][i];
                    }
                }
            });
    }

    /*Обратный ход*/
    for (int i = n - 1; i >= 0; i--) {
        x[i] = a[i][n] / a[i][i];
        for (int j = n - 1; j > i; j--) {
            x[i] = x[i] - a[i][j] * x[j] / a[i][i];
        }
    }
    cout << "Гаусс c распараллеливанием oneTBB: " << clock() - start << endl;
}

int main()
{
    setlocale(LC_ALL, "Russian");
    cout << "Введите размерность матрицы: " << endl;
    cin >> n;
    double** a = new double* [n];
    double* x = new double[n];
    double* x1 = new double[n];
    bool answer;
    for (int i = 0; i <= n; i++) {
        a[i] = new double[n + 1];
    }
    FillAArr(a, n);
    FillXArr(x1, n);
    FillBArr(a, x1, n);

    Gauss(a, x, n);
    answer = CheckAnswers(x1, x, n);
    if (answer)
        cout << "Ответ верный" << endl;
    else
        cout << "Ответ неверный" << endl;
    GaussParallel(a, x, n);
    GaussParallel2(a, x, n);

    system("pause");
    return 0;
}

#ifndef _I2C_H_
#define _I2C_H_

#include <unistd.h>
#include <linux/i2c-dev.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <cstdio>
#include <cstdlib>


template<class T>
class I2C {
private:
    int file;
public:
    I2C(const char *, int);

    ~I2C();

    bool send(const T &);

    bool request(T &);
};

template<class T>
I2C<T>::I2C(const char *filename, int addr) {
    if ((file = open(filename, O_RDWR)) < 0) {
        printf("Failed to open the bus.\n");
        /* ERROR HANDLING; you can check errno to see what went wrong */
        exit(1);
    }

    if (ioctl(file, I2C_SLAVE, addr) < 0) {
        printf("Failed to acquire bus access and/or talk to slave.\n");
        exit(1);
    }
}

template<class T>
I2C<T>::~I2C() {
    close(file);
}

template<class T>
bool I2C<T>::send(const T &data) {
    size_t size = sizeof data;
    auto writenSize = (size_t) write(file, (const unsigned char *) &data, size);
    return size == writenSize;
}

template<class T>
bool I2C<T>::request(T &data) {
    size_t size = sizeof data;
    auto readSize = (size_t) read(file, (unsigned char *) &data, size);
    return readSize == size;
}

#endif
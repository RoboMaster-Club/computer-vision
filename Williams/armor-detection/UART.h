#ifndef _UART_H_
#define _UART_H_

#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <termio.h>


template<class T>
class UART {
private:
    int fd;
    const char* fdname = "/dev/ttyTHS2";
    size_t size;
public:
    UART();

    ~UART();

    bool send(const T &);

    bool request(T &);
};

template<class T>
UART<T>::UART() {
    if ((fd = open(fdname, O_RDWR | O_NOCTTY | O_NDELAY)) < 0) {
        printf("Failed to open the bus.\n");
        //ERROR HANDLING; you can check errno to see what went wrong
        exit(1);
    }

    fcntl(fd, F_SETFL, 0);
    struct termios port_settings;               // structure to store the port settings in
    cfsetispeed(&port_settings, B115200);       // set baud rates
    cfsetospeed(&port_settings, B115200);

    port_settings.c_cflag &= ~PARENB;           // set no parity, stop bits, data bits
    port_settings.c_cflag &= ~CSTOPB;
    port_settings.c_cflag &= ~CSIZE;
    port_settings.c_cflag |= CS8;

    tcsetattr(fd, TCSANOW, &port_settings);     // apply the settings to the port

    size = sizeof(T);
}

template<class T>
UART<T>::~UART() {
    close(fd);
}

template<class T>
bool UART<T>::send(const T &data) {
    auto writenSize = (size_t) write(fd, (const unsigned char *) &data, size);
    return size == writenSize;
}

template<class T>
bool UART<T>::request(T &data) {
    auto readSize = (size_t) read(fd, (unsigned char *) &data, size);
    return readSize == size;
}

#endif

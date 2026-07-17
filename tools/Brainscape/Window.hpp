//
// Created by jaro on 7/17/26.
//

#ifndef YANN_WINDOW_HPP
#define YANN_WINDOW_HPP

#include <GLFW/glfw3.h>

class Window
{
public:
    Window(int width, int height, const char* title);
    ~Window();

    static void InitBackend();

    int getWidth();
    int getHeight();

    void setWidth(int width);
    void setHeight(int height);

    void setSize(int width, int height);

    void swapBuffers();
    void pollEvents();

    bool shouldClose();

    GLFWwindow* getWindow();
private:
    GLFWwindow* window;
    int width_, height_;
};

#endif //YANN_WINDOW_HPP

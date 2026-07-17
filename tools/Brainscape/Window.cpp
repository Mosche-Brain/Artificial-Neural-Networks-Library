//
// Created by jaro on 7/17/26.
//

#include "Window.hpp"

Window::Window(int width, int height, const char* title) : width_(width), height_(height)
{
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    glfwMakeContextCurrent(window);
}

Window::~Window()
{

}

void Window::InitBackend()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
}

int Window::getWidth()
{
    glfwGetWindowSize(window, &width_, nullptr);
    return width_;
}

int Window::getHeight()
{
    glfwGetWindowSize(window, nullptr, &height_);
    return height_;
}


void Window::setWidth(int width)
{
    glfwSetWindowSize(window, width, height_);
    width_ = width;
}

void Window::setHeight(int height)
{
    glfwSetWindowSize(window, width_, height);
    height_ = height;
}

void Window::setSize(int width, int height)
{
    glfwSetWindowSize(window, width, height);
    width_ = width;
    height_ = height;
}


void Window::swapBuffers()
{
    glfwSwapBuffers(window);
}

void Window::pollEvents()
{
    glfwPollEvents();
}

bool Window::shouldClose()
{
    return glfwWindowShouldClose(window);
}

GLFWwindow* Window::getWindow()
{
    return window;
}

#include "Application.hpp"

#include "Window.hpp"

#include <cum/cum.hpp>

int main()
{
    Window::InitBackend();
    cum::cum(cum::DEVICE::GPU);

    Application app;
    const int code = app.run();

    cum::decum();
    return code;
}

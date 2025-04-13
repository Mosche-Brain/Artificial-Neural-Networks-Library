template<typename Type>
struct vec2
{
    Type x;
    Type y;

    friend vec2 operator + (const vec2 &first, const vec2 &second)
    { 
        return vec2{first.x + second.x, first.y + second.y};
    }

    friend vec2 operator - (const vec2 &first, const vec2 &second)
    { 
        return vec2{first.x - second.x, first.y - second.y};
    }

    friend vec2 operator * (const vec2 &first, const vec2 &second)
    { 
        return vec2{first.x * second.x, first.y * second.y};
    }

    friend vec2 operator * (const vec2 &first, Type scalar)
    { 
        return vec2{first.x * second.x, first.y * second.y};
    }

    friend vec2 operator / (const vec2 &first, const vec2 &second)
    { 
        return vec2{first.x / second.x, first.y / second.y};
    }

    friend vec2 operator / (const vec2 &first, const Type scalar)
    { 
        return vec2{first.x / scalar, first.y / scalar};
    }

    friend vec2 operator ^ (const vec2 &first, const vec2 &second)
    { 
        return vec2{std::pow(first.x, second.x), std::pow(first.y, second.y)};
    }


};

template<typename Type>
struct vec3
{
    Type x;
    Type y;
    Type z;
};


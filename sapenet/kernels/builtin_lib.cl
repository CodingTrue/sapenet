#define DECL_FUNC(T1, T2, NAME, OPERATOR) \
inline void NAME( \
    const int G_ID, \
    T1 float* a, \
    T2 float* b, \
    global float* c, \
    const int a_offset, \
    const int b_offset, \
    const int c_offset, \
    const int size \
) { \
    if (G_ID >= size) return; \
    c[G_ID + c_offset] = a[G_ID + a_offset] OPERATOR b[G_ID + b_offset]; \
}
void add(
    const int X_ID,
    const int X_SIZE,
    memory_region float* a,
    memory_region float* b,
    global float* c,
    const int a_offset,
    const int b_offset,
    const int c_offset
) {
    if (X_ID >= X_SIZE) return;
    c[X_ID + c_offset] = a[X_ID + a_offset] + b[X_ID + b_offset];
}
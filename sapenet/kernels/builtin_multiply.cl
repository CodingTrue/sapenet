void multiply(
    const int G_ID,
    memory_region float* a,
    memory_region float* b,
    global float* c,
    const int a_offset,
    const int b_offset,
    const int c_offset,
    const int max_threads
) {
    if (G_ID >= max_threads) return;
    c[G_ID + c_offset] = a[G_ID + a_offset] * b[G_ID + b_offset];
}
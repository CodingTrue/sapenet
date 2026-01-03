void matmul(
    const int X_ID,
    const int Y_ID,
    const int X_SIZE,
    const int Y_SIZE,
    const int ROW_LENGTH,
    memory_region float* a,
    memory_region float* b,
    global float* c,
    const int a_offset,
    const int b_offset,
    const int c_offset
) {
    if (X_ID >= X_SIZE || Y_ID >= Y_SIZE) return;

    float sum = 0;
    for (int i = 0; i < ROW_LENGTH; i++) {
        sum += a[Y_ID * ROW_LENGTH + i + a_offset] * b[i * X_SIZE + X_ID + b_offset];
    }

    c[Y_ID * X_SIZE + X_ID + c_offset] = sum;
}
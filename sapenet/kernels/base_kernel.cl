kernel void compute_kernel(
    constant float* constant_data,
    global float* work_data
) {
    const int X_ID = get_global_id(0);
    const int Y_ID = get_global_id(1);

    const int X_SIZE = get_global_size(0);
    const int Y_SIZE = get_global_size(1);

    $BODY_SECTION
}
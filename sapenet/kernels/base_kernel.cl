kernel void compute_kernel(
    constant float* constant_data,
    global float* work_data
) {
    const int G_ID = get_global_id(0);
    $BODY_SECTION
}
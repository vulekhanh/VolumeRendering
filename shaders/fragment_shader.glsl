#version 120

uniform sampler3D u_volume;
uniform sampler2D u_transfer; // 1D transfer function stored in 2D
uniform vec3 u_vol_dims; // x,y,z dims
uniform mat4 u_inv_modelview;
uniform vec3 u_cam_pos; // camera position in model space
uniform float u_steps; // number of ray steps
uniform float u_opacity_scale;

varying vec2 v_texcoord;

const float EPS = 1e-3;

// sample transfer function: tf texture coordinate in [0,1]
vec4 sample_tf(float val) {
    return texture2D(u_transfer, vec2(val, 0.5));
}

// compute ray origin and direction in texture (0..1) space
void compute_ray(out vec3 ro, out vec3 rd) {
    // reconstruct position in NDC, then to model space
    vec2 ndc = v_texcoord * 2.0 - 1.0;
    vec4 nearPoint = vec4(ndc, -1.0, 1.0);
    vec4 farPoint  = vec4(ndc,  1.0, 1.0);

    vec4 worldNear = u_inv_modelview * nearPoint;
    vec4 worldFar  = u_inv_modelview * farPoint;
    worldNear /= worldNear.w;
    worldFar  /= worldFar.w;

    ro = worldNear.xyz * 0.5 + 0.5; // model->texture: assume model cube in [-1,1]
    vec3 ro_far = worldFar.xyz * 0.5 + 0.5;
    rd = normalize(ro_far - ro);
}

// ray-box intersect in [0,1]^3 texture space
bool intersect_box(vec3 ro, vec3 rd, out float tmin, out float tmax) {
    vec3 invR = 1.0 / (rd + vec3(1e-6));
    vec3 tbot = -ro * invR;
    vec3 ttop = (vec3(1.0) - ro) * invR;
    vec3 tmin3 = min(tbot, ttop);
    vec3 tmax3 = max(tbot, ttop);
    tmin = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    return tmax > tmin;
}

void main() {
    vec3 ro, rd;
    compute_ray(ro, rd);

    float t0, t1;
    if (!intersect_box(ro, rd, t0, t1)) {
        discard;
    }

    float t = t0;
    float dt = (t1 - t0) / u_steps;

    vec4 accum = vec4(0.0);
    for (int i = 0; i < 2000; ++i) { // hard upper bound to satisfy GLSL 1.2
        if (i >= int(u_steps)) break;
        vec3 pos = ro + rd * (t + 0.5 * dt);
        float sample = texture3D(u_volume, pos).r; // assume normalized 0..1
        vec4 col = sample_tf(sample);
        // pre-multiplied alpha compositing, front-to-back
        col.a *= u_opacity_scale;
        col.rgb *= col.a;
        accum.rgb = accum.rgb + (1.0 - accum.a) * col.rgb;
        accum.a = accum.a + (1.0 - accum.a) * col.a;
        if (accum.a >= 0.995) break; // early ray termination
        t += dt;
        if (t > t1) break;
    }

    // Apply gamma and output
    gl_FragColor = vec4(accum.rgb, accum.a);
}
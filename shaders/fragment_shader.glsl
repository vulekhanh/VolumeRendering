#version 120

uniform sampler3D u_volume;
uniform sampler2D u_transfer; // 1D transfer function stored in 2D
uniform vec3 u_vol_dims; // x,y,z dims (voxels)
uniform mat4 u_inv_modelview;
uniform vec3 u_cam_pos; // camera position in model space
uniform float u_steps; // number of ray steps
uniform float u_opacity_scale;

varying vec2 v_texcoord;

const float EPS = 1e-3;

// lighting uniforms
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform vec3 u_light_dir;

// early termination threshold (0..1)
uniform float u_terminate_thresh;

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

// sample volume value in [0,1]
float sample_volume(vec3 pos) {
    return texture3D(u_volume, pos).r;
}

// estimate gradient using central differences in texture space
vec3 estimate_gradient(vec3 pos) {
    // convert voxel sizes: step in texture coords = 1.0 / dims
    vec3 voxel = 1.0 / u_vol_dims;
    float gx = sample_volume(pos + vec3(voxel.x, 0.0, 0.0)) - sample_volume(pos - vec3(voxel.x, 0.0, 0.0));
    float gy = sample_volume(pos + vec3(0.0, voxel.y, 0.0)) - sample_volume(pos - vec3(0.0, voxel.y, 0.0));
    float gz = sample_volume(pos + vec3(0.0, 0.0, voxel.z)) - sample_volume(pos - vec3(0.0, 0.0, voxel.z));
    vec3 g = vec3(gx, gy, gz);
    float n = length(g);
    if (n > 1e-6) return normalize(g);
    return vec3(0.0, 0.0, 0.0);
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
    // front-to-back compositing
    for (int i = 0; i < 2000; ++i) { // GLSL 1.2 requires constant loop bounds
        if (i >= int(u_steps)) break;
        vec3 pos = ro + rd * (t + 0.5 * dt);

        // sample scalar
        float sample = sample_volume(pos);
        vec4 col = sample_tf(sample);

        // scale opacity by user control
        col.a *= u_opacity_scale;

        // shading: only if alpha significant
        if (col.a > 0.01) {
            // compute gradient normal in texture space
            vec3 N = estimate_gradient(pos);
            // fallback normal if too small
            if (length(N) < 1e-4) {
                N = vec3(0.0, 0.0, 1.0);
            }
            vec3 L = normalize(u_light_dir);
            float NdotL = max(dot(N, L), 0.0);

            // Blinn-Phong (half-vector)
            vec3 V = normalize(-rd);
            vec3 H = normalize(L + V);
            float spec = pow(max(dot(N, H), 0.0), u_shininess);

            vec3 shaded = u_ambient * col.rgb +
                          u_diffuse * NdotL * col.rgb +
                          u_specular * spec * vec3(1.0);
            col.rgb = shaded;
        }

        // pre-multiplied alpha compositing
        col.rgb *= col.a;
        accum.rgb = accum.rgb + (1.0 - accum.a) * col.rgb;
        accum.a = accum.a + (1.0 - accum.a) * col.a;

        // early termination: user-controlled threshold
        if (accum.a >= u_terminate_thresh) break;

        t += dt;
        if (t > t1) break;
    }

    gl_FragColor = vec4(accum.rgb, accum.a);
}

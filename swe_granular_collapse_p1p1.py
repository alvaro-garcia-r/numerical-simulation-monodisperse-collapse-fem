from dolfin import *
import numpy as np

# malla fina para alta resolución
nx = ny = 250
mesh = RectangleMesh(Point(-3, -3), Point(3, 3), nx, ny)

V = FunctionSpace(mesh, 'P', 1)

# parámetros físicos
g = 9.81
phi = 0.01 * np.pi / 180
mu = np.tan(phi)
eta = mu
epsilon = 1e-3
nu = 0.01
dt = 0.005
T = 3.0
num_steps = int(T / dt)

# funciones
h_n = Function(V)
u_n = Function(V)
v_n = Function(V)
h_np1 = Function(V)
u_np1 = Function(V)
v_np1 = Function(V)

# test functions
theta = TestFunction(V)
xi = TestFunction(V)
psi = TestFunction(V)

# condición inicial: gaussiana
h_max = 1.0
x_c, y_c = 0.0, 0.0
sigma = 0.3

h_init_expr = Expression(
    "h_max * exp(-((x[0] - xc)*(x[0] - xc) + (x[1] - yc)*(x[1] - yc)) / (2*sigma*sigma))",
    degree=2, h_max=h_max, xc=x_c, yc=y_c, sigma=sigma)
h_n.interpolate(h_init_expr)
u_n.assign(Constant(0.0))
v_n.assign(Constant(0.0))

# salida
xdmf_file = XDMFFile("swe_p1p1_250_phi_0/solucion.xdmf")
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.parameters["rewrite_function_mesh"] = False

# loop temporal
for n in range(num_steps):
    h_np1_dx = Dx(h_np1, 0)
    h_np1_dy = Dx(h_np1, 1)
    u_np1_dx = Dx(u_np1, 0)
    v_np1_dy = Dx(v_np1, 1)
    u_np1_dy = Dx(u_np1, 1)
    v_np1_dx = Dx(v_np1, 0)

    un_vec = as_vector([u_n, v_n])
    unorm_sq = inner(un_vec, un_vec)
    h_safe = conditional(gt(h_n, 1e-4), h_n, 1e-4)

    # F_h
    F_h = (1/dt)*inner(h_np1, theta)*dx \
        + inner(h_n*Dx(u_np1, 0), theta)*dx \
        + inner(u_n*h_np1_dx, theta)*dx \
        + inner(h_n*Dx(v_np1, 1), theta)*dx \
        + inner(v_n*h_np1_dy, theta)*dx \
        - (1/dt)*inner(h_n, theta)*dx

    # F_u con fricción + viscosidad
    F_u = (1/dt)*inner(h_n*u_np1, xi)*dx \
        + (1/dt)*inner(u_n*h_np1, xi)*dx \
        + 2*inner(h_n*u_n*u_np1_dx, xi)*dx \
        + inner(u_n*u_n*h_np1_dx, xi)*dx \
        + inner(h_n*u_n*v_np1_dy, xi)*dx \
        + inner(h_n*v_n*u_np1_dy, xi)*dx \
        + inner(u_n*v_n*h_np1_dy, xi)*dx \
        + g*inner(h_n*h_np1_dx, xi)*dx \
        + eta*g*inner(h_safe / sqrt(unorm_sq + epsilon) * u_np1, xi)*dx \
        + dt * nu * inner(grad(u_np1), grad(xi))*dx \
        - (2/dt)*inner(h_n*u_n, xi)*dx

    # F_v igual
    F_v = (1/dt)*inner(h_n*v_np1, psi)*dx \
        + (1/dt)*inner(v_n*h_np1, psi)*dx \
        + 2*inner(h_n*v_n*v_np1_dy, psi)*dx \
        + inner(v_n*v_n*h_np1_dy, psi)*dx \
        + inner(h_n*u_n*v_np1_dx, psi)*dx \
        + inner(h_n*v_n*u_np1_dx, psi)*dx \
        + inner(u_n*v_n*h_np1_dx, psi)*dx \
        + g*inner(h_n*h_np1_dy, psi)*dx \
        + eta*g*inner(h_safe / sqrt(unorm_sq + epsilon) * v_np1, psi)*dx \
        + dt * nu * inner(grad(v_np1), grad(psi))*dx \
        - (2/dt)*inner(h_n*v_n, psi)*dx

    solve(F_h == 0, h_np1)
    solve(F_u == 0, u_np1)
    solve(F_v == 0, v_np1)

    h_np1.rename("h", "Altura")
    u_np1.rename("u", "Vel_X")
    v_np1.rename("v", "Vel_Y")
    xdmf_file.write(h_np1, n * dt)
    xdmf_file.write(u_np1, n * dt)
    xdmf_file.write(v_np1, n * dt)

    h_n.assign(h_np1)
    u_n.assign(u_np1)
    v_n.assign(v_np1)

    if n % 20 == 0:
        print(f"paso {n}/{num_steps}, t = {n*dt:.3f} s")

xdmf_file.close()
print("simulación completada")

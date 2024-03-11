using CairoMakie

fig = Figure()
ax1 = Axis(fig[1,1])
CairoMakie.scatter!(ax1,1,1)
fig
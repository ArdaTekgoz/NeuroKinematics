import roboticstoolbox as rtb

# Puma560 robotunu yükle
robot = rtb.models.DH.Puma560()

print("Robot görselleştiriliyor... (Pencere açılmazsa arka plana bak)")

# Robotu "pyplot" (Matplotlib) motoruyla çizdir
# (Bu en basit görselleştiricidir, kurulum gerektirmez)
robot.plot(robot.qz, backend='pyplot', block=True)
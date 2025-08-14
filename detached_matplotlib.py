import os, getpass, socket
import matplotlib as mpl
mpl.use("WebAgg")
mpl.rcParams["webagg.open_in_browser"] = False
mpl.rcParams["webagg.address"] = os.environ.get("WEBAGG_ADDR", "127.0.0.1")
mpl.rcParams["webagg.port"] = int(os.environ.get("WEBAGG_PORT", "8988"))

host = mpl.rcParams["webagg.address"]; port = mpl.rcParams["webagg.port"]

# Start WebAgg server robustly and learn the real port in use
srv = None
try:
    from matplotlib.backends.backend_webagg import ServerThread
    srv = ServerThread(); srv.daemon = True; srv.start()
    # prefer server's bound port if it differs (e.g., due to retries)
    port = getattr(srv, "port", port)
except Exception:
    pass  # rely on the backend's own show() startup

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1,2,3],[4,5,6]); ax.set_title("WebAgg-safe demo")

# Register a manager now so the root page lists a figure
plt.show(block=False)

url = f"http://{host}:{port}/"
who = getpass.getuser()
fqdn = socket.getfqdn() or "remote.host"
print("Open in browser:", url)
print("If running on a remote box, tunnel with:")
print(f"ssh -N -L {port}:{host}:{port} {who}@{fqdn}")
print("Then open:", f"http://127.0.0.1:{port}/")

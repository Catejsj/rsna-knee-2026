import os, sys, glob
print("sys.path:", [p for p in sys.path if 'kaggle' in p or 'usr' in p])
for base in ("/kaggle/usr/lib", "/kaggle/lib", "/kaggle/input"):
    print(f"--- {base} ---")
    for p in sorted(glob.glob(base + "/*"))[:15]:
        print("  ", p)
        for q in sorted(glob.glob(p + "/*"))[:6]:
            print("      ", q)

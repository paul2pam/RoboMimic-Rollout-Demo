import sys
import h5py

path = sys.argv[1] if len(sys.argv) > 1 else 'path/to/your/lift_image.hdf5'
print(f"Opening: {path}\n")

f = h5py.File(path, 'r')

print("=== Top-level keys ===")
print(list(f.keys()))

if 'data' in f:
    print("\n=== data keys (first 10) ===")
    data_keys = list(f['data'].keys())
    print(data_keys[:10])

    if data_keys:
        first = data_keys[0]
        demo = f[f'data/{first}']
        print(f"\n=== data/{first} contents ===")
        demo.visit(lambda name: print(f"  {name}"))

        print(f"\n=== data/{first} dataset shapes ===")
        def print_shape(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        demo.visititems(print_shape)

        print(f"\n=== data/{first} attrs ===")
        print(dict(demo.attrs))
else:
    print("\n=== Full file structure ===")
    f.visit(lambda name: print(f"  {name}"))

    print("\n=== All dataset shapes ===")
    def print_shape(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
    f.visititems(print_shape)

f.close()

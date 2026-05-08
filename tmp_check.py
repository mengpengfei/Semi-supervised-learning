import sys
sys.path.insert(0, '/data/code/Semi-supervised-learning')

from collections import Counter
from semilearn.datasets.cv_datasets.cbcnet import CBCNetDataset
from torchvision.transforms import transforms
import math

# Create a simple transform
transform = transforms.Compose([
    transforms.Resize((int(math.floor(64 / 0.92)), int(math.floor(64 / 0.92)))),
    transforms.RandomCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Load dataset
dset = CBCNetDataset(
    alg='freematch',
    txt='/data2/fssd2/damagou_error/val.txt',
    num_classes=4364,
    transform=transform,
    is_ulb=False
)

print(f"Dataset size: {len(dset)}")
print(f"Mapping keys length: {len(dset.mapping_keys)}")

# Check first 100 samples
label_counter = Counter()
invalid_count = 0
max_label = -1
min_label = 999999

for i in range(min(1000, len(dset))):
    try:
        sample = dset[i]
        label = sample['y_lb']
        if label is not None:
            label_counter[label] += 1
            max_label = max(max_label, label)
            min_label = min(min_label, label)
    except Exception as e:
        invalid_count += 1
        if invalid_count <= 5:
            print(f"Error at index {i}: {e}")

print(f"\nValid samples in first 1000: {sum(label_counter.values())}")
print(f"Invalid samples: {invalid_count}")
print(f"Unique labels found: {len(label_counter)}")
print(f"Label range: [{min_label}, {max_label}]")
print(f"Expected range: [0, 4363]")

if max_label >= 4364:
    print("ERROR: Found label index >= num_classes!")
if min_label < 0:
    print("ERROR: Found negative label index!")

# Show top 10 most common labels
print("\nTop 10 most common labels:")
for label, count in label_counter.most_common(10):
    char = dset.mapping_keys[label] if label < len(dset.mapping_keys) else 'UNKNOWN'
    print(f"  Label {label} ('{char}'): {count} samples")
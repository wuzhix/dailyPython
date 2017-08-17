from PIL import Image
import imagehash


def make_regalur_image(img, size=(256, 256)):
    return img.resize(size).convert('RGB')


def split_image(img, part_size=(64, 64)):
    w, h = img.size
    pw, ph = part_size
    assert w % pw == h % ph == 0
    return [img.crop((i, j, i+pw, j+ph)).copy() \
                for i in range(0, w, pw) \
                for j in range(0, h, ph)]


def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)


def calc_similar(li, ri):
    return sum(hist_similar(l.histogram(), r.histogram()) for l, r in zip(split_image(li), split_image(ri))) / 16.0


def calc_similar_by_path(lf, rf):
    li, ri = make_regalur_image(Image.open(lf)), make_regalur_image(Image.open(rf))
    return calc_similar(li, ri)


img1 = Image.open('./../resource/1.jpg')
img2 = Image.open('./../resource/2.jpg')

# 通过phash计算图片相似，距离越小相似度越高
phash_a = imagehash.phash(img1)
phash_b = imagehash.phash(img2)
print(phash_a - phash_b)

# 通过ahash计算图片相似，距离越小相似度越高
ahash_a = imagehash.average_hash(img1)
ahash_b = imagehash.average_hash(img2)
print(ahash_a-ahash_b)

# 通过dhash计算图片相似，距离越小相似度越高
dhash_a = imagehash.dhash(img1)
dhash_b = imagehash.dhash(img2)
print(dhash_a-dhash_b)

# 将图像分块，根据小块直方图计算相似率
li, ri = make_regalur_image(img1), make_regalur_image(img2)
similar_rate = calc_similar(li, ri)
print(similar_rate)



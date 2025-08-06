import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']

# 可视化
def visualize_step(title, image, color_conversion=None):
    """可视化处理步骤的辅助函数"""
    if color_conversion:
        display_img = cv2.cvtColor(image, color_conversion)
    else:
        display_img = image.copy()

    # 调整显示尺寸
    h, w = display_img.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        display_img = cv2.resize(display_img, (int(w * scale), int(h * scale)))

    cv2.imshow(title, display_img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    return display_img

# 车牌定位
def locate_license_plate(image_path):
    # 1. 读取原始图像
    original = cv2.imread(image_path)
    if original is None:
        print("错误：无法读取图像，请检查路径")
        return None

    print("步骤1/8: 显示原始图像")

    # ____________________________________________________________________
    # vis_original = visualize_step("1. 显示原始图像".encode("gbk"), original)
    # ____________________________________________________________________

    # 2. 调整图像尺寸
    h, w = original.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img = cv2.resize(original, (int(w * scale), int(h * scale)))
        print(f"步骤2/8: 调整尺寸 (缩放比例: {scale:.2f})")

        # ____________________________________________________________________
        # vis_resized = visualize_step("2. 调整尺寸后的图像".encode("gbk"), img, cv2.COLOR_BGR2RGB)
        # ____________________________________________________________________

    else:
        img = original.copy()
        print("步骤2/8: 图像尺寸合适，无需调整")

    # 3. 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("步骤3/8: 转换到HSV颜色空间")

    # ____________________________________________________________________
    # vis_hsv = visualize_step("3. HSV颜色空间".encode("gbk"), hsv)
    # ____________________________________________________________________

    # 4. 创建颜色掩膜 (专注于蓝色车牌)
    # 优化蓝色车牌阈值
    blue_lower = np.array([95, 120, 80])  # 更宽的色调范围
    blue_upper = np.array([135, 255, 255])  # 包含更多蓝色变化

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    print("步骤4/8: 创建蓝色车牌掩膜")

    # ____________________________________________________________________
    # vis_blue_mask = visualize_step("4. 蓝色车牌掩膜".encode("gbk"), blue_mask)
    # ____________________________________________________________________

    # 5. 形态学操作
    # 优化核大小 (针对车牌字符)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))  # 更宽的核连接字符

    # 优化形态学操作序列
    # 先开运算去除小噪点
    opened = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # 再闭运算连接字符区域
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

    print("步骤5/8: 形态学操作")
    # 可视化核
    kernel_vis = cv2.resize(kernel * 255, (300, 75), interpolation=cv2.INTER_NEAREST)

    # ____________________________________________________________________

    # title_temp_5_1 = "5.1 形态学核 (20x5)".encode("gbk")
    # cv2.imshow(title_temp_5_1, kernel_vis)
    # cv2.waitKey(0)
    # cv2.destroyWindow(title_temp_5_1)

    # ____________________________________________________________________

    # ____________________________________________________________________
    # 可视化开运算结果
    # vis_opened = visualize_step("5.2 开运算结果".encode("gbk"), opened)
    # 可视化闭运算结果
    # vis_closed = visualize_step("5.3 闭运算结果".encode("gbk"), closed)
    # ____________________________________________________________________

    # 6. 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制所有轮廓
    contour_img = img.copy()

    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

    print("步骤6/8: 查找轮廓")
    # ________________________________________________________
    # vis_contours = visualize_step("6. 检测到的轮廓".encode("gbk"), contour_img, cv2.COLOR_BGR2RGB)
    # ________________________________________________________
    # 7. 筛选轮廓 - 针对蓝色车牌优化
    plate_contour = None
    max_area = 0
    candidate_info = []

    for i, cnt in enumerate(contours):
        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / max(h, 1)  # 防止除零
        area = w * h

        # 区域二次验证：字符密度检测
        roi = closed[y:y + h, x:x + w]
        white_pixels = cv2.countNonZero(roi)
        density = white_pixels / (w * h) if w * h > 0 else 0

        # 保存候选信息
        candidate_info.append({
            "id": i,
            "x": x, "y": y, "w": w, "h": h,
            "aspect_ratio": aspect_ratio,
            "area": area,
            "density": density,
            "valid": False,
            "reason": ""
        })

        # 关键筛选条件 - 针对蓝色车牌优化
        valid = True
        reason = ""

        # 面积筛选 - 排除太小区域
        if area < 1000:  # 最小面积1000像素
            valid = False
            reason = "Area too small"
        # 长宽比筛选 - 放宽条件
        elif aspect_ratio < 2.5 or aspect_ratio > 5.0:
            valid = False
            reason = f"Aspect ratio invalid({aspect_ratio:.1f})"
        # 密度筛选 - 放宽条件
        elif density < 0.2 or density > 0.8:
            valid = False
            # density = 白色像素数量 / 区域总像素数
            reason = f"Density invalid(Not a valid plate)({density:.2f})"

        if valid:
            candidate_info[-1]["valid"] = True
            candidate_info[-1]["reason"] = "valid area"

            # 保留最大有效区域
            if area > max_area:
                max_area = area
                plate_contour = cnt
                plate_rect = (x, y, w, h)
        else:
            candidate_info[-1]["reason"] = reason

    # 可视化候选轮廓
    candidate_img = img.copy()
    for i, cand in enumerate(candidate_info):
        color = (0, 255, 0) if cand["valid"] else (0, 0, 255)
        cv2.rectangle(candidate_img, (cand["x"], cand["y"]),
                      (cand["x"] + cand["w"], cand["y"] + cand["h"]), color, 2)

        # 添加标注信息
        label = f"ID:{cand['id']} AR:{cand['aspect_ratio']:.1f} D:{cand['density']:.2f}"
        cv2.putText(candidate_img, label, (cand["x"], cand["y"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 添加原因说明
        cv2.putText(candidate_img, cand["reason"], (cand["x"], cand["y"] + cand["h"] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    print("步骤7/8: 轮廓筛选 (绿色为有效候选)")

    # ________________________________________________________
    # vis_candidates = visualize_step("7. 轮廓筛选结果".encode("gbk"), candidate_img)
    final_result2_dir = "final_result2"
    os.makedirs(final_result2_dir, exist_ok=True)
    cv2.imwrite('final_result2/plate_located.png', candidate_img)

    # ________________________________________________________

    # 8. 显示最终结果
    if plate_contour is not None:
        result_img = img.copy()
        x, y, w, h = plate_rect

        # 绘制车牌边界框
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # 添加信息文本
        info_text = f"Plate: {w}x{h} pixels, Aspect: {w / h:.2f}, Density: {density:.2f}"
        cv2.putText(result_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        print("步骤8/8: 最终车牌定位")
        # ________________________________________________________
        # vis_result = visualize_step("8. 车牌定位结果".encode("gbk"), result_img, cv2.COLOR_BGR2RGB)
        # ________________________________________________________
        # 裁剪车牌区域
        plate_img = img[y:y + h, x:x + w].copy()

        # 显示车牌区域
        # title_temp_final = "定位的车牌区域".encode("gbk")
        # cv2.imshow(title_temp_final, plate_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite("located_plate.jpg", plate_img)
        print(f"车牌已保存为: located_plate.jpg")

        return plate_img
    else:
        print("未检测到车牌")
        print("可能原因:")
        print("1. 颜色阈值不匹配 - 尝试调整蓝色阈值范围")
        print("2. 轮廓筛选条件过严 - 检查筛选条件")
        print("3. 车牌区域不完整 - 检查原始图像质量")
        return None


def improved_preprocess_plate_image(plate_img):
    """
    改进的车牌预处理：增强分隔符处理
    """
    # 1. 动态计算尺寸相关参数
    height, width = plate_img.shape[:2]
    kernel_size = max(1, int(min(height, width) * 0.01))  # 基于图像尺寸的动态核大小

    # 2. 转换为灰度图并进行直方图均衡化
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 3. 自适应对比度增强 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 4. 自适应阈值二值化 - 动态块大小
    block_size = max(11, int(min(height, width) * 0.1) // 2 * 2 + 1)  # 确保为奇数
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,  # 使用均值法减少噪声影响
        cv2.THRESH_BINARY_INV,
        block_size,
        5
    )

    # 5. 噪声处理 - 动态核尺寸
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # 6. 智能边框检测
    # 水平投影
    horizontal_proj = np.sum(opened, axis=1)
    smoothed_hproj = scipy.signal.medfilt(horizontal_proj, kernel_size=5)  # 中值滤波平滑

    # 自适应阈值
    h_threshold = np.max(smoothed_hproj) * 0.1
    valid_rows = np.where(smoothed_hproj > h_threshold)[0]

    if len(valid_rows) > 0:
        top = valid_rows[0]
        bottom = valid_rows[-1]
    else:
        top, bottom = 0, height - 1

    # 垂直投影
    vertical_proj = np.sum(opened, axis=0)
    smoothed_vproj = scipy.signal.medfilt(vertical_proj, kernel_size=5)

    v_threshold = np.max(smoothed_vproj) * 0.05
    valid_cols = np.where(smoothed_vproj > v_threshold)[0]

    if len(valid_cols) > 0:
        left = valid_cols[0]
        right = valid_cols[-1]
    else:
        left, right = 0, width - 1

    # 裁剪车牌区域
    cropped_binary = opened[top:bottom + 1, left:right + 1]
    cropped_gray = enhanced[top:bottom + 1, left:right + 1]

    return cropped_binary, cropped_gray


def improved_split_characters(processed_img, original_img):
    """
    简化版字符分割：基于固定比例划分7个字符区域
    优化：第一个字符向右切割10%，最后一个字符向左切割10%
    """
    # 获取图像尺寸
    height, width = processed_img.shape[:2]

    # 定义字符区域比例
    total_width = width
    left_width = total_width * 0.25  # 前2字符区占25%
    right_width = total_width * 0.7  # 后5字符区占70%（预留5%分隔符位置）
    sep_x = left_width  # 分隔符位置

    # 左侧2字符
    left_regions = [
        {"x": left_width * 0.1, "y": 0, "w": left_width * 0.45, "h": height},  # 第一个字符（向右切割10%）
        {"x": left_width * 0.55, "y": 0, "w": left_width * 0.45, "h": height}  # 第二个字符
    ]

    # 右侧5字符，整体右移8%宽度
    shift_ratio = 0.08
    shift_amount = right_width * shift_ratio

    # 右侧5字符，最后一个字符向左切割10%
    right_regions = []
    for i in range(5):
        char_width = right_width / 5
        # 最后一个字符宽度减少10%
        if i == 4:
            char_width = char_width * 0.9

        right_regions.append({
            "x": sep_x + shift_amount + i * right_width / 5,
            "y": 0,
            "w": char_width,
            "h": height
        })

    # 合并所有区域
    valid_regions = left_regions + right_regions
    valid_regions = sorted(valid_regions, key=lambda r: r["x"])

    # 提取字符图像
    characters = []
    for i, region in enumerate(valid_regions):
        # 对第一个字符：进一步向右切割2%（少切一点避免无法识别文字）
        if i == 0:
            region["x"] += region["w"] * 0.02
            region["w"] *= 0.98

        # 对最后一个字符：进一步向左切割5%（避免右侧边框）
        if i == len(valid_regions) - 1:
            region["w"] *= 0.95

        # 计算边界
        start_x = int(max(0, region["x"]))
        end_x = int(min(width, region["x"] + region["w"]))

        # 提取字符区域
        char_img = original_img[:, start_x:end_x]

        # 上下裁剪：去除上下边缘的10%像素
        crop_height = int(char_img.shape[0] * 0.10)  # 计算裁剪高度（10%）
        char_img = char_img[crop_height:-crop_height, :]

        characters.append({
            "image": char_img,
            "position": (start_x, end_x)
        })

    # 可视化分割结果
    # plt.figure(figsize=(15, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(processed_img, cmap='gray')
    # plt.title('预处理后的车牌')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    comp_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    for region in valid_regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(comp_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # plt.imshow(comp_img)
    # plt.title(f'字符区域检测结果 ({len(valid_regions)}个字符)')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    return characters

# 对分割后的黑白字符车牌进一步去噪
def refine_character(char_img, extra):
    """
    字符精炼：生成32×48黑白二值图像
    优化：去除边缘白噪点（特别是铆钉引起的噪点）
    """
    # 确保图像是灰度图
    if len(char_img.shape) > 2:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

    # 1. OTSU二值化
    _, binary = cv2.threshold(char_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 2. 去除边缘白噪点（特别是铆钉引起的噪点）
    # 计算图像边界区域（边缘15%的区域）
    edge_width = int(binary.shape[1] * 0.15)
    edge_height = int(binary.shape[0] * 0.15)

    # 定义边缘掩码
    edge_mask = np.zeros_like(binary)

    # 上边缘
    edge_mask[:edge_height, :] = 1
    # 下边缘
    edge_mask[-edge_height:, :] = 1
    # 左边缘
    edge_mask[:, :edge_width] = 1
    # 右边缘
    edge_mask[:, -edge_width:] = 1

    # 查找边缘区域的小连通区域（可能是铆钉）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 遍历所有连通区域去除铆钉（只针对非首字符）
    if extra:

        for k in range(1, num_labels):

            x, y, w, h, area = stats[k]

            # 小区域检查（面积小于图像面积的15%）
            if area < (binary.shape[0] * binary.shape[1] * 0.15):
                # 检查是否在边缘区域
                region_mask = (labels == k).astype(np.uint8)
                edge_overlap = cv2.bitwise_and(region_mask, edge_mask)

                # 如果与边缘区域有重叠，可能是铆钉噪点
                if np.any(edge_overlap):
                    binary[labels == k] = 0  # 去除噪点

    else:

        for i in range(1, num_labels):

            x, y, w, h, area = stats[i]
            region_mask = (labels == i).astype(np.uint8)

            # 检查是否在边缘区域
            edge_overlap = cv2.bitwise_and(region_mask, edge_mask)

            if np.any(edge_overlap):
                # 条件2：细长白条（长宽比大于5且最小边小于3像素）
                if max(w, h) / min(w, h) > 5 and min(w, h) < 3:
                    # 额外验证：确保是线状结构（宽度或高度很小）
                    binary[labels == i] = 0  # 去除白条

    # 3. 查找非零像素确定字符边界
    non_zero = np.nonzero(binary)

    if len(non_zero[0]) == 0:
        # 无字符时返回空白图像
        return np.zeros((32, 48), dtype=np.uint8)

    # 计算字符边界
    top = np.min(non_zero[0])
    bottom = np.max(non_zero[0])
    left = np.min(non_zero[1])
    right = np.max(non_zero[1])

    # 4. 带边距裁剪字符
    pad = 2
    cropped = binary[
              max(0, top - pad):min(binary.shape[0], bottom + pad + 1),
              max(0, left - pad):min(binary.shape[1], right + pad + 1)
              ]
    # # 5. 直接缩放至32*48
    # if cropped.size == 0:
    #     return np.zeros((32, 48), dtype=np.uint8)
    #
    # resized = cv2.resize(cropped, (32, 48), interpolation=cv2.INTER_AREA)
    #
    # return resized

    # 5. 直接扩展画布到32×48（不缩放）
    # 在创建画布前添加
    max_height = 48
    max_width = 32

    # 创建目标尺寸画布（48高×32宽）
    result = np.zeros((48, 32), dtype=np.uint8)

    # 获取裁剪后图像的尺寸
    h, w = cropped.shape

    # 检查是否需要缩小
    if h > max_height or w > max_width:
        # 计算缩小比例
        scale = min(max_height / h, max_width / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 高质量缩小
        cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

        # 然后继续使用上面的放置逻辑

    # 计算放置位置（居中）
    start_y = (48 - h) // 2
    start_x = (32 - w) // 2

    # 确保不会越界
    start_y = max(0, start_y)
    start_x = max(0, start_x)

    # 计算实际可放置的高度和宽度
    place_h = min(h, 48 - start_y)
    place_w = min(w, 32 - start_x)

    # 将裁剪后的图像放置到画布中央
    result[start_y:start_y + place_h, start_x:start_x + place_w] = cropped[:place_h, :place_w]
    return result


def save_characters(characters, output_dir="characters"):
    """
    保存分割后的字符图像
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 清空目录
    for file in os.listdir(output_dir):
        if file.startswith("char_"):
            os.remove(os.path.join(output_dir, file))

    # 保存并可视化
    # plt.figure(figsize=(15, 3))
    for k, char_data in enumerate(characters):
        char_img = char_data["image"]
        # 只针对非首字符去除铆钉噪点,对首字符进行特殊处理
        if k != 0:
            refined = refine_character(char_img, True)
        else:
            refined = refine_character(char_img, False)
        # 保存精炼后的字符 (PNG格式)
        cv2.imwrite(os.path.join(output_dir, f"char_{k}.png"), refined)

        # 保存原始字符
        # cv2.imwrite(os.path.join(output_dir, f"char_{k}_raw.png"), char_img)

        # 可视化
        # plt.subplot(1, len(characters), k + 1)
        # plt.imshow(refined, cmap='gray')
        # plt.title(f'字符 {k}')
        # plt.axis('off')

    # plt.suptitle('分割后的字符')
    # plt.tight_layout()
    # plt.show()

    print(f"成功保存 {len(characters)} 个字符到目录: {output_dir}/")


def process_license_plate(plate_img):
    """
    处理车牌图像：分割字符并保存
    """
    # 1. 预处理车牌图像
    processed, gray_plate = improved_preprocess_plate_image(plate_img)

    # 2. 分割字符
    characters = improved_split_characters(processed, gray_plate)

    # 3. 保存字符
    save_characters(characters)

    return characters


def preprocess_image(image_path, img_size=(40, 32)):
    """
    预处理二值化黑白图片，使其符合模型输入要求
    :param image_path: 图片路径
    :param img_size: 图像目标尺寸 (高度, 宽度) -> (40, 32)
    :return: 预处理后的图像数组
    """
    # 读取图像 (直接以灰度模式读取)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 二值化处理 (确保是黑白二值图)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 调整尺寸到模型期望的大小 (高度, 宽度) = (40, 32)
    resized_img = cv2.resize(binary_img, (img_size[1], img_size[0]))

    # 归一化 (将像素值缩放到0-1范围)
    normalized_img = resized_img / 255.0

    # 添加通道维度 (灰度图只有1个通道)
    processed_img = np.expand_dims(normalized_img, axis=-1)  # 形状变为 (32, 40, 1)

    # 添加批次维度
    batched_img = np.expand_dims(processed_img, axis=0)  # 形状变为 (1, 32, 40, 1)

    return batched_img, resized_img


# ________________________________________
import tensorflow as tf

main_class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y', 'Z', '京', '闽', '粤', '苏', '沪', '浙']


# 3. 加载训练好的模型
def load_model(model_path):
    """加载保存的模型"""
    return tf.keras.models.load_model(model_path)


def predict_characters(model, characters_folder):
    """
    预测characters文件夹下的所有字符并在单张图中汇总结果
    :param model: 加载的模型
    :param characters_folder: 包含字符图像的文件夹路径
    """
    # 获取文件夹中的所有图像文件
    image_files = sorted([f for f in os.listdir(characters_folder)
                          if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    if not image_files:
        print(f"在文件夹 {characters_folder} 中未找到图像文件")
        return

    all_predictions = []
    all_confidences = []
    char_images = []

    print("=" * 60)
    print(f"开始预测文件夹 '{characters_folder}' 中的字符...")
    print("=" * 60)

    # 遍历并预测每个字符
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(characters_folder, img_file)

        # 预处理图像
        input_img, display_img = preprocess_image(img_path)

        # 进行预测
        predictions = model.predict(input_img)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # 获取预测结果
        predicted_label = main_class_names[predicted_class]

        # 存储结果
        all_predictions.append(predicted_label)
        all_confidences.append(confidence)
        char_images.append(display_img)

        # 打印当前字符的预测结果
        print(f"字符 {i + 1}/{len(image_files)}: {img_file} → {predicted_label} (置信度: {confidence:.2%})")

    # 创建汇总图像
    num_chars = len(image_files)
    fig = plt.figure(figsize=(max(12, num_chars * 2), 10))  # 增大尺寸以容纳额外图像

    # 使用GridSpec定义布局 (2行，2列)
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # 区域1: 轮廓筛选结果
    ax1 = fig.add_subplot(gs[0, 0])
    candidate_img = cv2.imread("final_result2/plate_located.png")
    candidate_img = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
    ax1.imshow(candidate_img)
    ax1.set_title("轮廓筛选结果", fontsize=12)
    ax1.axis('off')

    # 区域2: 字符识别结果标题
    ax2 = fig.add_subplot(gs[0, 1])
    final_string = "".join(all_predictions)
    avg_confidence = np.mean(all_confidences)
    ax2.text(0.5, 0.7, f"识别结果: {final_string}",
             fontsize=14, ha='center', va='center')
    ax2.text(0.5, 0.4, f"平均置信度: {avg_confidence:.2%}",
             fontsize=12, ha='center', va='center', color='gray')
    ax2.text(0.5, 0.1, f"字符数量: {num_chars}",
             fontsize=12, ha='center', va='center', color='blue')
    ax2.axis('off')

    # 区域3: 字符图像展示
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_axis_off()

    # 计算字符位置
    char_width = 1.0 / max(6, num_chars)

    # 显示所有字符图像
    for i in range(num_chars):
        left = i * char_width + char_width / 2 - 0.05
        ax_char = ax3.inset_axes([left, 0.3, char_width * 0.8, 0.6])
        ax_char.imshow(char_images[i], cmap='gray')
        ax_char.set_title(f"{all_predictions[i]}\n({all_confidences[i]:.2%})",
                          fontsize=10, y=-0.2)
        ax_char.axis('off')

    # 添加整体标题
    plt.suptitle("车牌识别完整流程结果", fontsize=16, y=0.95)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部标题留出空间

    # 显示结果
    plt.show()

    # 输出最终结果
    print("\n" + "=" * 60)
    print("最终识别结果:")
    print("=" * 60)
    print(f"识别字符串: {final_string}")
    print(f"平均置信度: {avg_confidence:.2%}")
    print("=" * 60)

    return final_string


# 5. 预测主函数
def main():
    # 加载模型 (替换为您的模型路径)
    MODEL_PATH = "models/best_main_model.h5"  # 例如: "models/char_classifier.h5"
    model = load_model(MODEL_PATH)

    # 打印模型摘要
    print("模型加载成功!")
    model.summary()

    # 打印输入形状信息
    input_shape = model.input_shape
    print(f"\n模型期望输入形状: {input_shape}")
    print(f"高度: {input_shape[1]}, 宽度: {input_shape[2]}, 通道数: {input_shape[3]}")

    predict_characters(model, "characters")


if __name__ == "__main__":
    # 改进思路：车牌的第一个字符一定是汉字，用汉字模型识别，车牌的第二个字符一定是字母，用字母模式识别，车牌之后的字符，用字母和数字模型识别

    # 替换为您的车牌图像路径
    image_path = "test4.png"

    # 执行车牌定位
    plate_img = locate_license_plate(image_path)

    if plate_img is not None:
        print("车牌定位成功!")
    else:
        print("车牌定位失败，请尝试调整参数或检查图像")

    # 加载定位好的车牌图像
    plate_img = cv2.imread("located_plate.jpg")  # 替换为您的车牌图像路径

    if plate_img is None:
        print("错误：无法读取车牌图像")
    else:
        # 显示原始车牌
        # plt.imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        # plt.title("原始车牌图像")
        # plt.axis('off')
        # plt.show()

        # 处理车牌
        characters = process_license_plate(plate_img)

        # 输出结果
        print(f"成功分割出 {len(characters)} 个字符:")
        for i, char in enumerate(characters):
            print(f"字符 {i}: 位置 {char['position']}, 尺寸 {char['image'].shape}")

    main()

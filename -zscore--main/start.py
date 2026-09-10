"""
股期配對交易系統 - 一鍵啟動器
==============================
功能：
1. 自動檢查並安裝所需套件
2. 增量更新股價資料到最新日期
3. 啟動 Streamlit 回測應用
"""

import subprocess
import sys
import os
import importlib

# 修正 Windows console 編碼問題
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 切換到腳本所在目錄
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


def check_and_install_packages():
    """檢查並安裝所需套件"""
    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'statsmodels': 'statsmodels',
        'yfinance': 'yfinance',
    }

    missing = []
    for import_name, pip_name in required.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[INSTALL] 安裝缺少的套件: {', '.join(missing)}")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet', *missing
        ])
        print("[OK] 套件安裝完成！")
    else:
        print("[OK] 所有套件已就緒")


def configure_streamlit():
    """設定 Streamlit 參數以避免詢問 Email"""
    streamlit_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
    os.makedirs(streamlit_dir, exist_ok=True)
    cred_file = os.path.join(streamlit_dir, "credentials.toml")
    if not os.path.exists(cred_file):
        with open(cred_file, "w", encoding="utf-8") as f:
            f.write('[general]\nemail = ""\n')


def update_stock_prices():
    """增量更新股價"""
    print("\n" + "=" * 60)
    print("  步驟 1/2：更新股價資料")
    print("=" * 60)

    update_script = os.path.join(SCRIPT_DIR, 'update_prices.py')
    if os.path.exists(update_script):
        result = subprocess.run(
            [sys.executable, update_script],
            cwd=SCRIPT_DIR
        )
        if result.returncode != 0:
            print("[WARN] 股價更新遇到問題，將使用現有資料繼續啟動。")
    else:
        print("[WARN] 找不到 update_prices.py，跳過更新。")


def launch_app():
    """啟動 Streamlit 應用"""
    print("\n" + "=" * 60)
    print("  步驟 2/2：啟動配對交易回測系統")
    print("=" * 60)

    app_file = os.path.join(SCRIPT_DIR, 'app.py')
    if not os.path.exists(app_file):
        print("[ERROR] 找不到 app.py！")
        return

    print("\n>>> 正在啟動 Streamlit 伺服器...")
    print("    開啟瀏覽器後即可使用系統")
    print("    按 Ctrl+C 可以關閉伺服器\n")

    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', app_file,
        '--browser.gatherUsageStats', 'false',
        '--theme.base', 'dark',
    ], cwd=SCRIPT_DIR)


def main():
    print()
    print("=" * 60)
    print("  台股股期配對交易回測系統 - 一鍵啟動")
    print("=" * 60)
    print("  1. 檢查環境套件")
    print("  2. 增量更新股價至最新日期")
    print("  3. 啟動 Streamlit 回測介面")
    print("=" * 60)
    print()

    try:
        # Step 0: 檢查套件
        check_and_install_packages()

        # Step 0.5: 設定 Streamlit (避免卡在詢問 Email)
        configure_streamlit()

        # Step 1: 更新股價
        update_stock_prices()

        # Step 2: 啟動應用
        launch_app()

    except KeyboardInterrupt:
        print("\n\n系統已關閉。")
    except Exception as e:
        print(f"\n[ERROR] 發生錯誤: {e}")
        input("\n按 Enter 關閉...")


if __name__ == '__main__':
    main()

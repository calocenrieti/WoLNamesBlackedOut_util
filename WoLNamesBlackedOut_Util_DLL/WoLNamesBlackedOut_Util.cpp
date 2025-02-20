// dllmain.cpp : DLL アプリケーションのエントリ ポイントを定義します。
#include "pch.h"
#include <iostream>
#include <vector>
#include <map>
#include <wrl/client.h>
#include <fstream>
#include <Windows.Storage.h>
#include <Windows.h>
#include <winrt/Windows.Storage.h>


//device_id 0のCUDAデバイスのCompute Capabilityを取得する関数
//majorx10+minorでCCを取得
extern "C" __declspec(dllexport) int GetCUDAComputeCapability() {

    int device = 0;
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, device);

    int cudacc = (deviceProp.major * 10 + deviceProp.minor);
    return cudacc;

}
//device_id 0のCUDAデバイス名を取得して、GTXが含まれていない場合はtrueを返す
//Tensorコアを持たないGeforce16x0GTXを落とす目的
 extern "C" __declspec(dllexport) bool GetRTXisEnable() {

    int device = 0;
    cudaDeviceProp deviceProp;
    // CUDAデバイスのプロパティを取得
    cudaGetDeviceProperties(&deviceProp, device);

    // デバイス名を取得
    std::string deviceName(deviceProp.name);

    // デバイス名に"GTX"が含まれているかをチェック
    if (deviceName.find("GTX") != std::string::npos) {
        return false; // "GTX"が含まれている場合
    } else {
        return true; // "GTX"が含まれていない場合
    }

}

//プライマリーディスプレイのGPUベンダーを取得する関数。返値はNVIDIAはN,AMDはA,IntelはI,それ以外はNULL
extern "C" __declspec(dllexport) char GetGpuVendor() {
    using namespace Microsoft::WRL;
    // ベンダーIDとメーカー名のマップを作成
    std::map<UINT, std::wstring> vendorMap = {
        {0x10DE, L"NVIDIA"},
        {0x1002, L"AMD"},
        {0x8086, L"Intel"}
    };

    // DXGIファクトリーの作成
    ComPtr<IDXGIFactory6> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        std::cerr << "Failed to create DXGI factory." << std::endl;
        return - 1;
    }

    // アダプターの列挙
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT adapterIndex = 0; factory->EnumAdapters1(adapterIndex, &adapter) != DXGI_ERROR_NOT_FOUND; ++adapterIndex) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        // プライマリアダプターを確認
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;  // ソフトウェアアダプターをスキップ
        }

        // ベンダーIDの取得と16進数形式での表示
        //std::wcout << L"Vendor ID: 0x" << std::hex << desc.VendorId << std::endl;

        // ベンダーIDに対応するメーカー名の取得と表示
        auto it = vendorMap.find(desc.VendorId);
        if (it != vendorMap.end()) {
            if (desc.VendorId == 0x8086) {
                if (wcsncmp(desc.Description, L"Intel(R) UHD Graphics 630", 24) == 0) {
                    return 'X';
                }
                return 'I';
            }
            return it->second[0];
        }
        else {
            return 'X';
        }
    }
}

//my_yolov8m.onnxファイルを読み込んでTensorRTのエンジンファイルmy_yolov8m.engineを出力する関数
extern "C" __declspec(dllexport) int onnx2trt()
{
    using namespace std;
    using namespace nvinfer1;
    using namespace Microsoft::WRL;
	using namespace winrt::Windows::Storage;

    class Logger : public ILogger
    {
        void log(Severity severity, const char* msg) noexcept override
        {
            // 情報レベルのメッセージを抑制する
            if (severity <= Severity::kWARNING)
                cout << msg << endl;
        }
    };
    // ILogger のインスタンス化
    Logger logger;

    // ローカルフォルダのパスを取得
    auto localFolder = ApplicationData::Current().LocalFolder();
    std::wstring localAppDataPath = localFolder.Path().c_str();

    // アプリケーション専用のフォルダパスを組み立てる
    std::wstring appFolderPath = std::wstring(localAppDataPath) + std::wstring { L"\\WoLNamesBlackedOut" }; 
    std::wstring engineFilePath = appFolderPath + std::wstring{ L"\\my_yolov8m.engine" };
    
    std::string engineFilePathStr(engineFilePath.length(), 0);
    std::transform(engineFilePath.begin(), engineFilePath.end(), engineFilePathStr.begin(), [](wchar_t c) {
        return (char)c;
        });
	const char* engineFilePathCStr = engineFilePathStr.c_str();

    // ビルダーを作成する
    auto builder = unique_ptr<IBuilder>(createInferBuilder(logger));

    // ネットワークを作成する（明示的なバッチ）
    uint32_t flag = 1U << static_cast<uint32_t>
        (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(flag));

    // ONNXパーサーを作成する: parser
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    // ファイルの読み取り
    const char* file_path = "my_yolov8m.onnx";
    parser->parseFromFile(file_path, static_cast<int32_t>(ILogger::Severity::kWARNING));

    // trtがモデルを最適化する方法を指定するビルド構成を作成する
    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    // 設定の構成
    // ワークスペースのサイズ
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    // 精度の設定
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // 最適化構成を作成して設定する
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("images", OptProfileSelector::kMIN, Dims4{ 1, 3, 1280, 1280 });
    profile->setDimensions("images", OptProfileSelector::kOPT, Dims4{ 8, 3, 1280, 1280 });
    profile->setDimensions("images", OptProfileSelector::kMAX, Dims4{ 16, 3, 1280, 1280 });
    config->addOptimizationProfile(profile);

    // クリエーションエンジン
    auto engine = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    //シリアル化保存エンジン
    ofstream engine_file(engineFilePathCStr, ios::binary);
    //assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char*)engine->data(), engine->size());
    engine_file.close();

    cout << "Engine build success!" << endl;
    return 0;
}
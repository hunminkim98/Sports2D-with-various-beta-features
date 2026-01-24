
[![Continuous integration](https://github.com/davidpagnon/sports2d/actions/workflows/continuous-integration.yml/badge.svg?branch=main)](https://github.com/davidpagnon/sports2d/actions/workflows/continuous-integration.yml)
[![PyPI version](https://badge.fury.io/py/Sports2D.svg)](https://badge.fury.io/py/Sports2D)\
[![Downloads](https://static.pepy.tech/badge/sports2d)](https://pepy.tech/project/sports2d)
[![Stars](https://img.shields.io/github/stars/davidpagnon/sports2d)](https://github.com/davidpagnon/sports2d/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/davidpagnon/sports2d)](https://github.com/davidpagnon/sports2d/issues)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/davidpagnon/sports2d)](https://GitHub.com/davidpagnon/sports2d/issues?q=is%3Aissue+is%3Aclosed)\
[![status](https://joss.theoj.org/papers/1d525bbb2695c88c6ebbf2297bd35897/status.svg)](https://joss.theoj.org/papers/1d525bbb2695c88c6ebbf2297bd35897)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10576574.svg)](https://zenodo.org/doi/10.5281/zenodo.7903962)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)\
[![Discord](https://img.shields.io/discord/1183750225471492206?logo=Discord&label=Discord%20community)](https://discord.com/invite/4mXUdSFjmt)
[![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Sports2D-yellow?logo=huggingface)](https://huggingface.co/spaces/DavidPagnon/sports2d)


<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/Sports2D_Colab)-->



# Sports2D

**`Sports2D`는 비디오 또는 웹캠에서 2D 관절 위치, 관절 각도 및 세그먼트 각도를 자동으로 계산합니다.**

</br>

> **`공지사항:`**
> - 바닥 각도, 바닥 높이, 깊이 원근 효과 보정, 캘리브레이션 파일 생성 **v0.8.25 신규 기능!**
> - 분석하고 싶은 사람만 선택 가능 **v0.8 신규 기능!**
> - OpenSim을 통한 MarkerAugmentation 및 역기구학으로 정확한 3D 동작 분석 **v0.7 신규 기능!**
> - 모든 검출기 및 포즈 추정 모델 사용 가능 **v0.6 신규 기능!**
> - 픽셀이 아닌 미터 단위 결과 **v0.5 신규 기능!**
> - 더 빠르고 정확함
> - 웹캠에서 작동
> - 개선된 시각화 출력
> - 더 유연하고 실행이 쉬움
>
> 최신 버전을 받으려면 `pip install sports2d pose2sim -U`를 실행하세요.

***참고:*** 언제나 기여를 환영합니다 ([기여 방법](#기여-방법-및-할-일-목록) 참조)!
<!--사용자 친화적인 Colab 버전 출시! (최신 이슈도 수정됨)\
모든 스마트폰에서 작동!**\
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/Sports2D_Colab)-->


</br>


https://github.com/user-attachments/assets/2ce62012-f28c-4e23-b3b8-f68931bacb77

<!-- https://github.com/user-attachments/assets/6a444474-4df1-4134-af0c-e9746fa433ad -->

<!-- https://github.com/user-attachments/assets/1c6e2d6b-d0cf-4165-864e-d9f01c0b8a0e -->

`경고:` 각도 추정은 포즈 추정 알고리즘의 정확도에 따라 달라지며, 완벽하지 않습니다.\
`경고:` 피험자가 2D 평면(시상면 또는 전두면)에서 움직일 때만 결과가 정확합니다. 피험자는 동작 평면에 최대한 평행하게 촬영되어야 합니다.\
연구 수준의 3D 마커리스 관절 운동학이 필요하다면 여러 카메라를 사용하는 **[Pose2Sim](https://github.com/perfanalytics/pose2sim)**을 고려하세요.

<!--`경고:` Google Colab은 데이터 프라이버시에 관한 유럽 GDPR 요구사항을 따르지 않습니다. 이것이 중요하다면 [로컬 설치](#설치)를 하세요.-->

<!--`알려진 문제`: 일부 iPhone 세로 모드 동영상에서는 결과가 좋지 않을 수 있습니다(Colab에서 작업하는 경우 제외). `ffmpeg -i video_input.mov video_output.mp4`로 미리 변환하거나, https://video-converter.com 같은 온라인 비디오 변환기를 사용하면 해결됩니다.-->


## 목차
1. [설치 및 데모](#설치-및-데모)
   1. [Hugging Face에서 테스트](#hugging-face에서-테스트)
   1. [로컬 설치](#로컬-설치)
      1. [빠른 설치](#빠른-설치)
      2. [전체 설치](#전체-설치)
      3. [SynthPose 설치 (선택사항)](#synthpose-설치-선택사항)
   2. [데모](#데모)
      1. [데모 실행](#데모-실행)
      2. [OpenSim에서 시각화](#opensim에서-시각화)
      3. [Blender에서 시각화](#blender에서-시각화)
2. [매개변수 활용](#매개변수-활용)
   1. [사용자 비디오 또는 웹캠에서 실행](#사용자-비디오-또는-웹캠에서-실행)
   2. [특정 시간 범위에서 실행](#특정-시간-범위에서-실행)
   3. [관심 있는 사람 선택](#관심-있는-사람-선택)
   4. [미터 단위 좌표 얻기](#미터-단위-좌표-얻기)
   5. [역기구학 실행](#역기구학-실행)
   6. [여러 비디오 한 번에 실행](#여러-비디오-한-번에-실행)
   7. [설정 파일 사용 또는 Python 내에서 실행](#설정-파일-사용-또는-python-내에서-실행)
   8. [원하는 방식으로 각도 얻기](#원하는-방식으로-각도-얻기)
   9. [출력 사용자 정의](#출력-사용자-정의)
   10. [사용자 정의 포즈 추정 모델 사용](#사용자-정의-포즈-추정-모델-사용)
   11. [SynthPose 백엔드 사용](#synthpose-백엔드-사용)
   12. [모든 매개변수](#모든-매개변수)
3. [더 알아보기](#더-알아보기)
   1. [너무 느린가요?](#너무-느린가요)
   3. [역기구학 실행](#역기구학-실행)
   4. [작동 원리](#작동-원리)
4. [인용 방법 및 기여 방법](#인용-방법-및-기여-방법)

<br>

## 설치 및 데모


### Hugging Face에서 테스트

제한된 온라인 버전을 [Hugging Face](https://huggingface.co/spaces/DavidPagnon/sports2d)에서 테스트하세요: [![Hugging Face Space](https://img.shields.io/badge/HuggingFace-Sports2D-yellow?logo=huggingface)](https://huggingface.co/spaces/DavidPagnon/sports2d)

<img src="Content/huggingface_demo.png" width="760">



### 로컬 설치

<!--- 옵션 0: **Colab 사용** \
  사용자 친화적(전체) 버전, 휴대폰이나 태블릿에서도 작동.\
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/Sports2D_Colab)\
  YouTube 튜토리얼:\
  <a href = "https://www.youtube.com/watch?v=Er5RpcJ8o1Y"><img src="Content/Video_tuto_Sports2D_Colab.png" width="380"></a>

-->

#### 빠른 설치

> 참고: OpenSim 역기구학을 사용하려면 전체 설치가 필요합니다.

터미널을 엽니다. `python -V`를 입력하여 python >=3.10 <=3.12가 설치되어 있는지 확인하세요. 설치되어 있지 않다면 [여기서](https://www.python.org/downloads/) 설치하세요.

실행:
``` cmd
pip install sports2d
```

또는 소스에서 빌드하여 최신 변경사항을 테스트:
``` cmd
git clone https://github.com/davidpagnon/sports2d.git
cd sports2d
pip install .
```

<br>

#### 전체 설치

> **참고:** 역기구학을 실행하려는 경우에만 필요합니다 (`--do_ik True`).\
> **참고:** 이미 Pose2Sim conda 환경이 있다면 이 단계를 건너뛸 수 있습니다. `conda activate Pose2Sim`과 `pip install sports2d`만 실행하면 됩니다.

- Anaconda 또는 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 설치:\
  Anaconda 프롬프트를 열고 가상 환경을 생성:
  ``` cmd
  conda create -n Sports2D python=3.12 -y
  conda activate Sports2D
  ```
- **OpenSim 설치**:\
  OpenSim Python API 설치 (conda를 통해 설치하지 않으려면 [이 페이지](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python#ScriptinginPython-SettingupyourPythonscriptingenvironment(ifnotusingconda)) 참조):
    ```
    conda install -c opensim-org opensim -y
    ```

- **Sports2D와 Pose2Sim 설치**:
  ``` cmd
  pip install sports2d
  ```

<br>

#### SynthPose 설치 (선택사항)

> **참고:** SynthPose는 Stanford MIMI의 VitPose 모델을 사용하는 대체 포즈 추정 백엔드입니다.
> **중요:** SynthPose는 **PyTorch 의존성이 필요**합니다. RTMLib(기본 백엔드)보다 설치 용량이 크지만, 52개의 키포인트(17개 COCO + 35개 해부학적 마커)를 제공합니다.

SynthPose를 사용하려면 추가 의존성을 설치해야 합니다:

```cmd
pip install sports2d[synthpose]
```

또는 수동으로 PyTorch와 Transformers 설치:
```cmd
pip install torch torchvision transformers
```

**GPU 가속 (권장):**

CUDA를 사용하는 NVIDIA GPU의 경우:
```cmd
# CUDA 버전에 맞는 PyTorch 설치 (예: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

macOS의 경우 (Apple Silicon):
```cmd
# MPS 가속이 자동으로 감지됩니다
pip install torch torchvision
```

**SynthPose vs RTMLib 비교:**

| 특성 | RTMLib (기본) | SynthPose |
|------|--------------|-----------|
| 의존성 | ONNX Runtime (가벼움) | PyTorch + Transformers (~2GB+) |
| 키포인트 수 | 26개 (HALPE_26) | 52개 (17 COCO + 35 해부학적) |
| 모델 | RTMPose | VitPose (huge/base) |
| GPU 지원 | ONNX 프로바이더 | CUDA/MPS/CPU |
| 속도 | 빠름 | 정확도 우선 |


<br>

### 데모

#### 데모 실행:

명령줄을 열고 실행:
``` cmd
sports2d
```

관절 위치와 각도가 실시간으로 표시됩니다.

명령줄을 실행한 폴더에서 결과 `video`, `images`, `TRC pose` 및 `MOT angle` 파일(스프레드시트 소프트웨어로 열 수 있음), 그리고 `logs`를 찾을 수 있습니다.

***중요:*** conda 설치를 실행했다면 먼저 환경을 활성화해야 합니다: Anaconda 프롬프트에서 `conda activate sports2d`를 실행하세요.

<img src="Content/Demo_results.png" width="760">
<img src="Content/Demo_plots.png" width="760">
<img src="Content/Demo_terminal.png" width="760">

***참고:***\
데모 비디오는 정렬, 보간 및 필터링 후 프로세스의 견고성을 보여주기 위해 의도적으로 어렵게 만들어졌습니다. 포함 내용:
- 시상면에서 걷는 한 사람
- 전두면에서 점핑잭을 하는 한 사람. 이 사람은 역광 상태에서 플립을 수행하며, 둘 다 포즈 검출 알고리즘에 어려움
- 무시해야 하는 배경에서 깜빡이는 작은 사람

<br>


#### Blender에서 시각화

1. **Pose2Sim_Blender 애드온 설치.**\
   [Pose2Sim_Blender](https://github.com/davidpagnon/Pose2Sim_Blender) 애드온 페이지의 지침을 따르세요.
2. **카메라와 비디오 가져오기.**
    - **Cameras -> Import**: `result_dir` 폴더에서 `demo_calib.toml` 파일을 엽니다.
    - **Images/Videos -> Show**: 비디오 파일(예: `demo_Sports2D.mp4`)을 엽니다.\
    -> **Other tools -> See through camera**
2. **포인트 좌표 열기.**\
   **OpenSim data -> Markers**: `result_dir` 폴더에서 trc 파일(예: `demo_Sports2D_m_person00.trc`)을 엽니다.\
   이렇게 하면 캡처된 사람의 동작을 기반으로 **애니메이션 리그**가 선택적으로 생성됩니다.
3. **애니메이션 스켈레톤 열기:**\
   먼저 `--do_ik True`를 설정했는지 확인하세요 ([전체 설치](#전체-설치) 필요). 자세한 내용은 [역기구학](#역기구학-실행) 섹션을 참조하세요.
   - **OpenSim data -> Model**: 스케일된 모델(예: `demo_Sports2D_m_person00_LSTM.osim`)을 엽니다.
   - **OpenSim data -> Motion**: 모션 파일(예: `demo_Sports2D_m_person00_LSTM_ik.mot`)을 엽니다.

   OpenSim 스켈레톤은 아직 리깅되지 않았습니다. **[기여를 환영합니다!](https://github.com/perfanalytics/pose2sim/issues/40)** [![Discord](https://img.shields.io/discord/1183750225471492206?logo=Discord&label=Discord%20community)](https://discord.com/invite/4mXUdSFjmt)

<img src="Content/sports2d_blender.gif" width="760">

<br>


#### OpenSim에서 시각화

1. **[OpenSim GUI](https://simtk.org/frs/index.php?group_id=91)** 설치.
2. **포인트 좌표 시각화:**\
   **File -> Preview experimental data:** `result_dir` 폴더에서 trc 파일(예: `coords_m.trc`)을 엽니다.
3. **각도 시각화:**\
   애니메이션 모델을 열고 추가 생체역학 분석을 실행하려면 먼저 `--do_ik True`를 설정했는지 확인하세요 ([전체 설치](#전체-설치) 필요). 자세한 내용은 [역기구학](#역기구학-실행) 섹션을 참조하세요.
   - **File -> Open Model:** 스케일된 모델(예: `Model_Pose2Sim_LSTM.osim`)을 엽니다.
   - **File -> Load Motion:** 모션 파일(예: `angles.mot`)을 엽니다.

<img src="Content/sports2d_opensim.gif" width="760">

<br>



### 매개변수 활용

사용 가능한 모든 매개변수의 전체 목록은 문서의 [이 섹션](#모든-매개변수)을 참조하거나, [Config_Demo.toml](https://github.com/davidpagnon/Sports2D/blob/main/Sports2D/Demo/Config_demo.toml) 파일을 확인하거나, `sports2d --help`를 입력하세요. 지정되지 않은 모든 것은 기본값으로 설정됩니다.

<br>


#### 사용자 비디오 또는 웹캠에서 실행:
``` cmd
sports2d --video_input path_to_video.mp4
```

``` cmd
sports2d --video_input webcam
```

<br>

#### 특정 시간 범위에서 실행:
```cmd
sports2d --time_range 1.2 2.7
```

<br>


#### 관심 있는 사람 선택:
검출된 사람 중 일부만 분석하려면 `--nb_persons_to_detect`와 `--person_ordering_method` 매개변수를 사용할 수 있습니다. [미터 단위 좌표 변환](#미터-단위-좌표-얻기)이나 [역기구학 실행](#역기구학-실행)을 원한다면 순서가 중요합니다.


``` cmd
sports2d --nb_persons_to_detect 2 --person_ordering_method highest_likelihood
```

수동 입력이 가능하다면 `on_click` 방식을 권장합니다. 이를 통해 사람 수와 순서를 같은 단계에서 처리할 수 있습니다. 프롬프트가 표시되면 원하는 순서로 관심 있는 사람을 선택하세요. 예를 들어, 두 사람이 모두 보이는 프레임으로 이동하여 여성을 먼저 선택하고 남성을 선택합니다.

그렇지 않으면 Sports2D를 자동으로 실행하려는 경우 'highest_likelihood', 'largest_size', 'smallest_size', 'greatest_displacement', 'least_displacement', 'first_detected', 'last_detected' 같은 다른 정렬 방법을 선택할 수 있습니다.

``` cmd
sports2d --person_ordering_method on_click
```



<img src="Content/Person_selection.png" width="760">


<br>


#### 미터 단위 좌표 얻기:
> **참고:** Z 좌표(깊이)는 과도하게 신뢰해서는 안 됩니다.

픽셀을 미터로 변환하려면 최소한 참가자의 키가 필요합니다. 깊이 정보도 제공하면 더 나은 결과를 얻을 수 있습니다. 카메라 수평 각도와 바닥 높이는 일반적으로 자동으로 추정됩니다. **참고: 캘리브레이션 파일이 생성됩니다.**

- 픽셀-미터 스케일은 참가자의 미터 단위 키와 픽셀 단위 키의 비율로 계산됩니다. 픽셀 단위 키는 자동으로 계산되며, `--first_person_height` 매개변수를 사용하여 미터 단위 키를 지정하세요.
- 깊이 원근 효과는 카메라-피험자 거리(m), 초점 거리(px), 화각(도 또는 라디안), 또는 캘리브레이션 파일로 보정할 수 있습니다. `--perspective_unit` ('distance_m', 'f_px', 'fov_deg', 'fov_rad', 또는 'from_calib')와 `--perspective_value` 매개변수를 사용하세요.
- 카메라 수평 각도는 운동학에서 자동 추정(`auto`), 캘리브레이션 파일에서(`from_calib`), 또는 수동(float)으로 설정할 수 있습니다. `--floor_angle` 매개변수를 사용하세요.
- 바닥 레벨도 마찬가지입니다. `--xy_origin` 매개변수를 사용하세요.

이러한 매개변수 중 하나가 `from_calib`로 설정된 경우 `--calib_file`을 사용하세요.


``` cmd
sports2d --first_person_height 1.65
```
``` cmd
sports2d --first_person_height 1.65 `
        --floor_angle auto `
        --xy_origin auto`
        --perspective_unit distance_m --perspective_value 10
```
``` cmd
sports2d --first_person_height 1.65 `
        --floor_angle 0 `
        --xy_origin from_calib`
        --perspective_unit from_calib --calib_file Sports2D\Demo\Calib_demo.toml
```
``` cmd
sports2d --first_person_height 1.65 `
        --perspective_unit f_px --perspective_value 2520
```

<br>


#### 역기구학 실행:
> 참고: [전체 설치](#전체-설치)가 필요합니다.

> **참고:** 피험자는 선택한 전체 시간 범위 동안 단일 평면에서 움직여야 합니다.

OpenSim 역기구학을 사용하면 관절 제약, 관절 각도 제한을 설정하고, 뼈가 전체 동작 동안 같은 길이를 유지하도록 제약하며, 잠재적으로 좌우 크기를 동일하게 할 수 있습니다. 일반적으로 생체역학적으로 더 정확한 결과를 제공합니다. 또한 [MoCo](https://opensim-org.github.io/opensim-moco-site/)를 사용하여 관절 토크, 근육 힘, 지면 반력 등을 계산할 수 있습니다.

이것은 [Pose2Sim](https://github.com/perfanalytics/pose2sim)을 통해 수행됩니다.\
모델 스케일링은 프레임 하위 집합에 걸친 세그먼트 길이의 평균에 따라 수행됩니다. 10% 가장 빠른 프레임(잠재적 이상치), 속도가 0인 프레임(사람이 프레임 밖에 있을 가능성), 평균 무릎 및 엉덩이 굴곡 각도가 45° 이상인 프레임(사람이 웅크릴 때 포즈 추정이 정확하지 않음), 이전 작업 후 가장 극단적인 세그먼트 값의 20%(잠재적 이상치)를 제거합니다. 이러한 모든 매개변수는 Config.toml 파일에서 편집할 수 있습니다.

**참고: 피험자가 단일 평면에서 움직이지 않는 구간에서는 작동하지 않습니다. 필요한 경우 비디오를 여러 시간 범위로 분할할 수 있습니다.**

```cmd
sports2d --time_range 1.2 2.7 `
         --do_ik true --first_person_height 1.65 --visible_side auto front
```

선택적으로 LSTM 마커 증강을 사용하여 출력 모션의 품질을 향상시킬 수 있습니다.\
참가자에게 적절한 질량을 선택적으로 부여할 수도 있습니다. 질량은 운동(모션)에 영향을 미치지 않고 힘(추후 운동역학 분석을 진행하는 경우)에만 영향을 미칩니다.\
선택적으로 [Blender에서 오버레이된 결과를 시각화](#blender에서-시각화)할 수도 있습니다. 자동 캘리브레이션은 이렇게 짧은 시간 범위에서는 정확하지 않으므로 제공된 캘리브레이션 파일(또는 전체 걷기에서 생성된 파일)을 사용해야 합니다.

```cmd
sports2d --time_range 1.2 2.7 `
         --do_ik true --first_person_height 1.65 --visible_side left front `
         --use_augmentation True --participant_mass 55.0 67.0 `
         --calib_file Calib_demo.toml
```

<br>


#### 여러 비디오 한 번에 실행:
``` cmd
sports2d --video_input demo.mp4 other_video.mp4
```
모든 비디오가 같은 시간 범위로 분석됩니다.
```cmd
sports2d --video_input demo.mp4 other_video.mp4 --time_range 1.2 2.7
```
각 비디오에 다른 시간 범위.
```cmd
sports2d --video_input demo.mp4 other_video.mp4 --time_range 1.2 2.7 0 3.5
```

<br>


#### 설정 파일 사용 또는 Python 내에서 실행:

- 설정 파일로 실행:
  ``` cmd
  sports2d --config Config_demo.toml
  ```
- Python 내에서 실행, 예:\
  - `Demo/Config_demo.toml`을 편집하고 실행:
    ```python
    from Sports2D import Sports2D
    from pathlib import Path
    import toml

    config_path = Path(Sports2D.__file__).parent / 'Demo'/'Config_demo.toml'
    config_dict = toml.load(config_path)
    Sports2D.process(config_dict)
    ```
  - 또는 기본값이 아닌 값만 전달할 수 있습니다:
    ```python
    from Sports2D import Sports2D
    config_dict = {
      'base': {
        'nb_persons_to_detect': 1,
        'person_ordering_method': 'greatest_displacement'
        },
      'pose': {
        'mode': 'lightweight',
        'det_frequency': 50
        }}
    Sports2D.process(config_dict)
    ```

<br>


#### 원하는 방식으로 각도 얻기:

- 필요한 각도 선택:
  ```cmd
  sports2d --joint_angles 'right knee' 'left knee' --segment_angles None
  ```
- 각도 표시 위치 선택: 이미지 왼쪽 상단에 목록으로, 관절/세그먼트 근처에, 또는 둘 다:
  ```cmd
  sports2d --display_angle_values_on body # 또는 none, 또는 list
  ```
- 각도를 계산하고 표시하지 않도록 결정할 수도 있습니다:
  ```cmd
  sports2d --calculate_angles false
  ```
- 사람이 다른 쪽을 향할 때 각도 뒤집기.\
  **참고: 스프린트 시 false로 설정하세요.** *발끝 키포인트가 발뒤꿈치 오른쪽에 있으면 각 사지가 "오른쪽을 보는" 것으로 간주합니다. 이것이 항상 맞는 것은 아니며, 특히 스프린트의 스윙 단계에서 그렇습니다. 참가자가 자세를 바꿔도 시계열이 연속되길 원한다면 false로 설정하세요.*
  ```cmd
  sports2d --flip_left_right true # 기본값
  ```
- 추정된 카메라 틸트 각도에 따라 세그먼트 각도 보정.\
  **참고:** *카메라 틸트 각도는 자동으로 추정됩니다. 카메라가 아닌 바닥이 기울어진 경우 false로 설정하세요.*
  ```cmd
  sports2d --correct_segment_angles_with_floor_angle true # 기본값
  ```

- **OpenSim으로 역기구학**을 실행하려면 [이 섹션](#역기구학-실행)을 확인하세요

<br>


#### 출력 사용자 정의:
- 비디오, 이미지, trc 포즈 파일, 각도 mot 파일, 실시간 표시, 그래프를 원하는지 선택:
  ```cmd
  sports2d --save_vid false --save_img true `
           --save_pose false --save_angles true `
           --show_realtime_results false --show_graphs false
  ```
- 결과를 사용자 정의 디렉토리에 저장, 슬로우 모션 팩터 지정:
  ``` cmd
  sports2d --result_dir path_to_result_dir
  ```

<br>


#### 사용자 정의 포즈 추정 모델 사용:
- 손 동작 검색:
  ``` cmd
  sports2d --pose_model whole_body
  ```
- 모든 사용자 정의(배포된) MMPose 모델 사용
  ``` cmd
  sports2d --pose_model BodyWithFeet : `
           --mode """{'det_class':'YOLOX', `
                  'det_model':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', `
                  'det_input_size':[640, 640], `
                  'pose_class':'RTMPose', `
                  'pose_model':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip', `
                  'pose_input_size':[192,256]}"""
  ```

<br>


#### SynthPose 백엔드 사용

SynthPose는 Stanford MIMI의 VitPose 모델을 사용하는 대체 포즈 추정 백엔드입니다. 52개의 키포인트(17개 COCO + 35개 해부학적 마커)를 제공합니다.

> **중요:** SynthPose는 PyTorch 의존성이 필요합니다. 먼저 [SynthPose 설치](#synthpose-설치-선택사항)를 참조하세요.

기본 사용법:
```cmd
sports2d --pose_model synthpose
```

VitPose 모델 선택:
```cmd
# VitPose-huge (가장 정확, 느림) - 기본값
sports2d --pose_model synthpose --mode performance

# VitPose-base (속도/정확도 균형)
sports2d --pose_model synthpose --mode balanced

# 또는 명시적으로 base 모델 지정
sports2d --pose_model synthpose_base
```

검출기 선택:
```cmd
# YOLOX (권장 - 빠르고 안정적)
sports2d --pose_model synthpose --synthpose_detector yolox

# RT-DETR (HuggingFace, 좋은 정확도)
sports2d --pose_model synthpose --synthpose_detector rtdetr

# RT-DETRv4 (로컬 엔진 필요)
sports2d --pose_model synthpose --synthpose_detector rtdetrv4
```

Python에서 사용:
```python
from Sports2D import Sports2D

config_dict = {
    'pose': {
        'pose_model': 'synthpose',
        'mode': 'performance',  # 'performance', 'balanced', 또는 'lightweight'
        'device': 'auto',  # 'auto', 'cuda', 'mps', 또는 'cpu'
        'synthpose_detector': 'yolox'
    }
}
Sports2D.process(config_dict)
```

**SynthPose 모드 매핑:**

| 모드 | VitPose 모델 | 설명 |
|------|-------------|------|
| `performance` | VitPose-huge | 가장 정확, 느림 |
| `balanced` | VitPose-base | 속도/정확도 균형 |
| `lightweight` | VitPose-base + 경고 | SynthPose는 lightweight 모드 미지원 |

<br>


#### 모든 매개변수

사용 가능한 모든 매개변수의 전체 목록은 [Config_Demo.toml](https://github.com/davidpagnon/Sports2D/blob/main/Sports2D/Demo/Config_demo.toml) 파일을 확인하거나 다음을 입력하세요:

``` cmd
sports2d --help
```

```
'config': ["C", "toml 설정 파일 경로"],

'video_input': ["i", "webcam, 또는 video_path.mp4, 또는 video1_path.avi video2_path.mp4 ... 경로에 ASCII가 아닌 문자가 포함되면 이미지가 저장되지 않음"],
'time_range': ["t", "start_time end_time. 초 단위. 지정하지 않으면 전체 비디오. 다른 시간 범위의 여러 비디오인 경우 start_time1 end_time1 start_time2 end_time2 ..."],
'nb_persons_to_detect': ["n", "검출할 사람 수. 정수 또는 'all'. 지정하지 않으면 'all'"],
'person_ordering_method': ["", "'on_click', 'highest_likelihood', 'largest_size', 'smallest_size', 'greatest_displacement', 'least_displacement', 'first_detected', 또는 'last_detected'. 지정하지 않으면 'on_click'"],
'first_person_height': ["H", "참조 인물의 미터 단위 키. 지정하지 않으면 1.65. 캘리브레이션 파일이 제공되면 사용되지 않음"],
'visible_side': ["", "front, back, left, right, auto, 또는 none. 지정하지 않으면 'auto front none'. 'auto'면 동작 방향에 따라 left 또는 right. 'none'이면 이 사람에 대해 IK 없음"],
'participant_mass': ["", "참가자 질량(kg) 또는 none. 제공되지 않으면 기본값 70. 운동학(모션)에 영향 없음, 운동역학(힘)에만 영향"],
'perspective_value': ["", "카메라-피험자 거리(m), 초점 거리(px), 화각(도 또는 라디안), 또는 perspective_unit=='from_calib'이면 ''"],
'perspective_unit': ["", "'distance_m', 'f_px', 'fov_deg', 'fov_rad', 또는 'from_calib'"],
'do_ik': ["", "역기구학 수행. 지정하지 않으면 false"],
'use_augmentation': ["", "LSTM 마커 증강 사용. 지정하지 않으면 false"],
'load_trc_px': ["", "포즈 추정을 다시 실행하지 않고 trc 파일 로드. 지정하지 않으면 false"],
'compare': ["", "trc 파일과 모션을 시각적으로 비교. 지정하지 않으면 false"],
'video_dir': ["d", "지정하지 않으면 현재 디렉토리"],
'result_dir': ["r", "지정하지 않으면 현재 디렉토리"],
'webcam_id': ["w", "웹캠 ID. 지정하지 않으면 0"],
'show_realtime_results': ["R", "실시간 결과 표시. 지정하지 않으면 true"],
'display_angle_values_on': ["a", '"body", "list", "body" "list", 또는 "none". 지정하지 않으면 body list'],
'show_graphs': ["G", "원시 및 처리된 결과의 플롯 표시. 지정하지 않으면 true"],
'save_graphs': ["", "원시 및 처리된 결과의 위치 및 각도 플롯 저장. 지정하지 않으면 true"],
'joint_angles': ["j", '지정하지 않으면 "Right ankle" "Left ankle" "Right knee" "Left knee" "Right hip" "Left hip" "Right shoulder" "Left shoulder" "Right elbow" "Left elbow"'],
'segment_angles': ["s", '지정하지 않으면 "Right foot" "Left foot" "Right shank" "Left shank" "Right thigh" "Left thigh" "Pelvis" "Trunk" "Shoulders" "Head" "Right arm" "Left arm" "Right forearm" "Left forearm"'],
'save_vid': ["V", "처리된 비디오 저장. 지정하지 않으면 true"],
'save_img': ["I", "처리된 이미지 저장. 지정하지 않으면 true"],
'save_pose': ["P", "포즈를 trc 파일로 저장. 지정하지 않으면 true"],
'calculate_angles': ["c", "관절 및 세그먼트 각도 계산. 지정하지 않으면 true"],
'save_angles': ["A", "각도를 mot 파일로 저장. 지정하지 않으면 true"],
'slowmo_factor': ["", "슬로우 모션 팩터. 240 fps로 녹화하고 30 fps로 내보낸 비디오의 경우 240/30 = 8. 지정하지 않으면 1"],
'pose_model': ["p", "body_with_feet, whole_body_wrist, whole_body, body, synthpose, 또는 synthpose_base. 지정하지 않으면 body_with_feet"],
'mode': ["m", 'light, balanced, performance, 또는 """삼중 따옴표 내 딕셔너리""". 지정하지 않으면 balanced. 딕셔너리를 사용하여 자체 검출 및/또는 포즈 추정 모델 지정(문서에서 자세한 내용).'],
'det_frequency': ["f", "N 프레임마다 사람 검출 실행, 그 사이에는 이전에 검출된 바운딩 박스 추적. 키포인트 검출은 여전히 모든 프레임에서 실행.\n\
                  1 이상, 단순하고 혼잡하지 않은 경우 원하는 만큼 높게 설정 가능. 훨씬 빠르지만 정확도가 떨어질 수 있음. 지정하지 않으면 1: 모든 프레임에서 검출 실행"],
'backend': ["", "포즈 추정 백엔드는 'auto', 'cpu', 'cuda', 'mps'(macOS용), 또는 'rocm'(AMD GPU용)"],
'device': ["", "포즈 추정 장치는 'auto', 'openvino', 'onnxruntime', 'opencv'"],
'synthpose_detector': ["", "SynthPose 검출기: 'yolox'(권장), 'rtdetr', 또는 'rtdetrv4'. 지정하지 않으면 'yolox'"],
'to_meters': ["M", "픽셀을 미터로 변환. 지정하지 않으면 true"],
'make_c3d': ["", "trc를 c3d 파일로 변환. 지정하지 않으면 true"],
'floor_angle': ["", "바닥 각도(도). 지정하지 않으면 'auto'"],
'xy_origin': ["", "xy 평면의 원점. 지정하지 않으면 'auto'"],
'calib_file': ["", "캘리브레이션 파일 경로. 지정하지 않으면 '', 즉 캘리브레이션 파일 없음"],
'save_calib': ["", "캘리브레이션 파일 저장. 지정하지 않으면 true"],
'feet_on_floor': ["", "발이 바닥 레벨에 있도록 마커 증강 결과 오프셋. 지정하지 않으면 true"],
'distortions': ["", "카메라 왜곡 계수 [k1, k2, p1, p2, k3] 또는 'from_calib'. 지정하지 않으면 [0.0, 0.0, 0.0, 0.0, 0.0]"],
'use_simple_model': ["", "IK 10배 이상 빠름, 하지만 근육이나 유연한 척추, 슬개골 없음. 지정하지 않으면 false"],
'close_to_zero_speed_m': ["","모든 키포인트 합계: 약 50 px/frame 또는 0.2 m/frame"],
'tracking_mode': ["", "'sports2d' 또는 'deepsort'. 'deepsort'는 느리고 매개변수화하기 어렵지만 올바르게 조정하면 더 견고할 수 있음"],
'deepsort_params': ["", 'Deepsort 추적 매개변수: """삼중 따옴표 내 딕셔너리""". \n\
                    기본값: max_age:30, n_init:3, nms_max_overlap:0.8, max_cosine_distance:0.3, nn_budget:200, max_iou_distance:0.8, embedder_gpu: True\n\
                    자세한 정보: https://github.com/levan92/deep_sort_realtime/blob/master/deep_sort_realtime/deepsort_tracker.py#L51'],
'input_size': ["", "width, height. 지정하지 않으면 1280, 720. 낮은 해상도는 빠르지만 정밀도가 떨어짐"],
'keypoint_likelihood_threshold': ["", "검출된 키포인트는 likelihood가 이 임계값 미만이면 유지되지 않음. 지정하지 않으면 0.3"],
'average_likelihood_threshold': ["", "검출된 사람은 평균 키포인트 likelihood가 이 임계값 미만이면 유지되지 않음. 지정하지 않으면 0.5"],
'keypoint_number_threshold': ["", "검출된 사람은 검출된 키포인트 수가 이 임계값 미만이면 유지되지 않음. 지정하지 않으면 0.3, 즉 30 퍼센트"],
'max_distance': ["", "사람이 이전 프레임의 위치에서 max_distance보다 멀리 검출되면 새로운 사람으로 간주. px 또는 None, 기본값 100."],
'fastest_frames_to_remove_percent': ["", "속도가 빠른 프레임은 이상치로 간주. 기본값 0.1"],
'close_to_zero_speed_px': ["", "모든 키포인트 합계: 약 50 px/frame 또는 0.2 m/frame. 기본값 50"],
'large_hip_knee_angles': ["", "이 값 미만의 엉덩이 및 무릎 각도는 부정확한 것으로 간주. 기본값 45"],
'trimmed_extrema_percent': ["", "평균 계산 전 제거할 가장 극단적인 세그먼트 값의 비율. 기본값 50"],
'fontSize': ["", "각도 값의 글꼴 크기. 지정하지 않으면 0.3"],
'flip_left_right': ["", "true 또는 false. 사람이 다른 쪽을 향할 때 각도를 뒤집음. 발끝 키포인트가 발뒤꿈치 오른쪽에 있으면 사람이 오른쪽을 봄. 스프린트 중이거나 참가자가 자세를 바꿔도 시계열이 연속되길 원하면 false로 설정. 지정하지 않으면 true"],
'correct_segment_angles_with_floor_angle': ["", "true 또는 false. 카메라가 기울어진 경우 바닥 각도에 따라 세그먼트 각도를 보정. 카메라가 아닌 바닥이 기울어진 경우 false로 설정. 지정하지 않으면 True"],
'interpolate': ["", "누락된 데이터 보간. 지정하지 않으면 true"],
'interp_gap_smaller_than': ["", "N 프레임 미만의 누락 데이터 시퀀스 보간. 지정하지 않으면 10"],
'fill_large_gaps_with': ["", "last_value, nan, 또는 zeros. 지정하지 않으면 last_value"],
'sections_to_keep': ["", "all, largest, first, 또는 last. 미검출 청크가 산재해 있어도 'all' 유효 섹션 유지, 또는 'largest' 유효 섹션, 또는 'first', 또는 'last'"],
'min_chunk_size': ["", "사람의 데이터 청크를 유지하기 위한 연속 유효 프레임의 최소 수. 지정하지 않으면 10"],
'reject_outliers': ["", "다른 필터링 방법 전에 Hampel 필터로 이상치 제거. 지정하지 않으면 true"],
'filter': ["", "결과 필터링. 지정하지 않으면 true"],
'filter_type': ["", "butterworth, kalman, gcv_spline, gaussian, median, 또는 loess. 지정하지 않으면 butterworth"],
'cut_off_frequency': ["", "Butterworth 필터의 차단 주파수. 지정하지 않으면 6"],
'order': ["", "Butterworth 필터의 차수. 지정하지 않으면 4"],
'gcv_cut_off_frequency': ["", "GCV 스플라인 필터의 차단 주파수. 'auto'가 보통 더 좋음, 신호가 너무 짧으면 노이즈가 신호로 간주될 수 있음 -> 궤적이 필터링되지 않음. 지정하지 않으면 'auto'"],
'gcv_smoothing_factor': ["", "GCV 스플라인 필터의 스무딩 팩터(>=0). cut_off_frequency != 'auto'이면 무시. 더 많은 스무딩(>1) 또는 데이터에 더 충실(<1). 지정하지 않으면 1.0"],
'trust_ratio': ["", "Kalman 필터의 신뢰 비율: 등가속도 가정(프로세스)보다 삼각측량 결과(측정)를 얼마나 더 신뢰하는가? 지정하지 않으면 500"],
'smooth': ["", "듀얼 Kalman 스무딩. 지정하지 않으면 true"],
'sigma_kernel': ["", "가우시안 필터의 시그마. 지정하지 않으면 1"],
'nb_values_used': ["", "loess 필터에 사용되는 값의 수. 지정하지 않으면 5"],
'kernel_size': ["", "미디언 필터의 커널 크기. 지정하지 않으면 3"],
'butterspeed_order': ["", "속도에 대한 Butterworth 필터의 차수. 지정하지 않으면 4"],
'butterspeed_cut_off_frequency': ["", "속도에 대한 Butterworth 필터의 차단 주파수. 지정하지 않으면 6"],
'osim_setup_path': ["", "OpenSim 설정 경로. 지정하지 않으면 '../OpenSim_setup'"],
'right_left_symmetry': ["", "좌우 대칭. 지정하지 않으면 true"],
'default_height': ["", "스케일링을 위한 기본 키. 지정하지 않으면 1.70"],
'remove_individual_scaling_setup': ["", "스케일링 중 생성된 개별 스케일링 설정 파일 제거. 지정하지 않으면 true"],
'remove_individual_ik_setup': ["", "IK 중 생성된 개별 IK 설정 파일 제거. 지정하지 않으면 true"],
'fastest_frames_to_remove_percent': ["", "속도가 빠른 프레임은 이상치로 간주. 기본값 0.1"],
'close_to_zero_speed_m': ["","모든 키포인트 합계: 약 0.2 m/frame. 기본값 0.2"],
'close_to_zero_speed_px': ["", "모든 키포인트 합계: 약 50 px/frame. 기본값 50"],
'large_hip_knee_angles': ["", "이 값 미만의 엉덩이 및 무릎 각도는 부정확한 것으로 간주되어 무시됨. 기본값 45"],
'trimmed_extrema_percent': ["", "평균 계산 전 제거할 가장 극단적인 세그먼트 값의 비율. 기본값 50"],
'use_custom_logging': ["", "사용자 정의 로깅 사용. 지정하지 않으면 false"]
```

<br>


## 더 알아보기

### 너무 느린가요?

**빠른 수정:**
- ` --save_vid false --save_img false --show_realtime_results false` 사용: 이미지나 비디오를 저장하지 않고 실시간 결과를 표시하지 않음.
- `--mode lightweight` 사용: RTMPose의 더 가벼운 버전을 사용하여 빠르지만 정확도가 떨어짐.\
참고로 모든 검출 및 포즈 모델을 사용할 수 있습니다(.onnx 또는 .zip 파일이 없다면 먼저 [MMPose로 배포](https://mmpose.readthedocs.io/en/latest/user_guides/how_to_deploy.html#onnx)):
  ```
  --mode """{'det_class':'YOLOX',
          'det_model':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_nano_8xb8-300e_humanart-40f6f0d0.zip',
          'det_input_size':[416,416],
          'pose_class':'RTMPose',
          'pose_model':'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
          'pose_input_size':[192,256]}"""
  ```
- `--det_frequency 50` 사용: Rtmlib는 (기본적으로) 탑다운 방식: 프레임의 모든 사람에 대한 바운딩 박스를 검출한 다음 각 박스 내의 키포인트를 검출. 사람 검출 단계가 훨씬 느림. 50 프레임마다만 사람을 검출하고 그 사이에는 바운딩 박스를 추적하도록 선택할 수 있어 훨씬 빠름.
- `--load_trc_px <path_to_file_px.trc>` 사용: 파일에서 포즈 추정 결과를 사용. 검출 및 포즈 추정을 모두 다시 실행하지 않고 픽셀-미터 변환이나 각도 계산에 다른 매개변수를 사용하려는 경우 유용.
- `--tracking_mode sports2d` 사용: 기본 Sports2D 트래커를 사용. DeepSort와 달리 더 빠르고 매개변수화가 필요 없으며 혼잡하지 않은 장면에서 동등하게 좋음.

<br>

**GPU 사용**:\
정확도에 영향 없이 훨씬 빠름. 하지만 설치에 약 6GB의 추가 저장 공간이 필요.

1. 터미널에서 `nvidia-smi`를 실행하세요. 오류가 발생하면 GPU가 CUDA와 호환되지 않을 가능성이 높습니다. 아니라면 "CUDA version"을 확인하세요: 드라이버와 호환되는 최신 버전입니다 ([이 포스트](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with)에서 자세한 정보).

   그런 다음 [ONNXruntime 요구사항 페이지](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)로 가서 최신 호환 CUDA 및 cuDNN 요구사항을 확인하세요. 다음으로 [pyTorch 웹사이트](https://pytorch.org/get-started/previous-versions/)로 가서 이 요구사항을 충족하는 최신 버전을 설치하세요 (torch 2.4는 cuDNN 9와 함께 제공되지만 torch 2.3은 cuDNN 8을 설치합니다). 예:
   ``` cmd
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

<!-- > ***참고:*** 기본 명령에서 문제가 보고되었습니다. 하지만 이것은 테스트되어 작동합니다:
`pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118` -->

2. 마지막으로 GPU 지원으로 ONNX Runtime을 설치:
   ```
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

3. Python에서 다음 명령으로 모든 것이 제대로 되었는지 확인:
   ``` bash
   python -c 'import torch; print(torch.cuda.is_available())'
   python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
   # "True ['CUDAExecutionProvider', ...]"가 출력되어야 함
   ```
   <!-- print(f'torch version: {torch.__version__}, cuda version: {torch.version.cuda}, cudnn version: {torch.backends.cudnn.version()}, onnxruntime version: {ort.__version__}') -->

<br>






<!--

여기에 비디오

-->


<br>




### 작동 원리

Sports2D:
- RTMLib 또는 SynthPose를 사용하여 비디오나 웹캠에서 2D 관절 중심을 검출합니다.
- 픽셀 좌표를 미터로 변환합니다.
- 선택된 관절 및 세그먼트 각도를 계산합니다.
- 선택적으로 OpenSim을 통해 운동학적 최적화를 수행합니다.
- 선택적으로 처리된 이미지와 비디오 파일을 저장합니다.

<br>

**정확히 어떻게 작동하나요?**\
Sports2D:

1. **웹캠, 하나의 비디오, 또는 비디오 목록에서 스트림을 읽습니다**. 처리할 지정된 시간 범위를 선택합니다.

2. **RTMLib 또는 SynthPose로 포즈 추정을 설정합니다.** lightweight, balanced, 또는 performance 모드로 실행할 수 있으며, 더 빠른 추론을 위해 사람 바운딩 박스를 매 프레임 검출 대신 추적할 수 있습니다. 모든 RTMPose 또는 SynthPose 모델을 사용할 수 있습니다.

3. **사람을 추적**하여 ID가 프레임 간에 일관되도록 합니다. 사람은 작은 거리에 있을 때 다음 프레임의 다른 사람과 연결됩니다. 'sports2d' 트래커 덕분에 사람이 몇 프레임에서 사라져도 ID가 일관되게 유지됩니다. [v0.8.22 릴리스 노트에서 자세한 정보 참조](https://github.com/davidpagnon/Sports2D/releases/tag/v0.8.22).

4. **분석할 사람을 선택합니다.** 단일 사람 모드에서는 시퀀스에서 가장 높은 평균 점수를 가진 사람만 유지합니다. 다중 사람 모드에서는 분석할 사람 수(`nb_persons_to_detect`)와 정렬 방법(`person_ordering_method`)을 선택할 수 있습니다. 정렬 방법은 'on_click', 'highest_likelihood', 'largest_size', 'smallest_size', 'greatest_displacement', 'least_displacement', 'first_detected', 'last_detected'가 될 수 있습니다. `on_click`이 기본값이며 사용자가 원하는 순서로 관심 있는 사람을 클릭할 수 있게 합니다.

4. **픽셀 좌표를 미터로 변환합니다.** 사용자는 지정된 사람의 크기를 제공하여 결과를 그에 맞게 스케일할 수 있습니다. 카메라 수평 각도와 바닥 레벨은 보행 시퀀스에서 자동으로 검출하거나 수동으로 지정하거나 캘리브레이션 파일에서 얻을 수 있습니다. 깊이 원근 효과는 피험자까지의 카메라 거리, 초점 거리, 화각, 또는 캘리브레이션 파일로 보정됩니다. [v0.8.25 릴리스 노트에서 자세한 정보 참조](https://github.com/davidpagnon/Sports2D/releases/tag/v0.8.25).

5. **선택된 관절 및 세그먼트 각도를 계산**하고 해당 발이 왼쪽/오른쪽을 가리키면 좌우로 뒤집습니다.

5. **이미지에 결과를 그립니다:**\
  각 사람 주위에 바운딩 박스를 그리고 ID를 씁니다\
  스켈레톤과 키포인트를 그리며 신뢰도에 따라 녹색에서 빨간색 색상 스케일 사용\
  관절 및 세그먼트 각도를 몸에 그리고 값을 관절/세그먼트 근처 또는 이미지 왼쪽 상단에 진행 막대와 함께 씁니다

6. **결과를 보간하고 필터링합니다:** (1) 좌우 사지 간 스왑을 보정, (2) 갭이 너무 크지 않으면 누락된 포즈와 각도 시퀀스를 보간, (3) Hampel 필터로 이상치를 제거, 마지막으로 (4) 결과를 필터링, 기본적으로 6 Hz Butterworth 필터 사용. 위의 모든 것은 설정하거나 비활성화할 수 있으며, Kalman, GCV, Gaussian, LOESS, Median, 속도에 대한 Butterworth 같은 다른 필터도 사용 가능 ([Config_Demo.toml](https://github.com/davidpagnon/Sports2D/blob/main/Sports2D/Demo/Config_demo.toml) 참조)

7. **선택적으로** 처리된 이미지 표시, 저장, 또는 비디오로 저장\
  **선택적으로** 비교를 위해 처리 전후의 포즈 및 각도 데이터 플롯\
  **선택적으로** 각 사람의 포즈를 픽셀 및 미터 단위의 TRC 파일로, 각도를 MOT 파일로, 캘리브레이션 데이터를 [Pose2Sim](https://github.com/perfanalytics/pose2sim) TOML 파일로 저장

8. **선택적으로 [Pose2Sim](https://github.com/perfanalytics/pose2sim)을 통해 OpenSim으로 스케일링 및 역기구학을 실행합니다.**

<br>

**관절 각도 규약:**
- 발목 배굴곡: 발뒤꿈치와 엄지발가락, 발목과 무릎 사이.\
  *발이 정강이와 일직선일 때 -90°.*
- 무릎 굴곡: 엉덩이, 무릎, 발목 사이.\
  *정강이가 허벅지와 일직선일 때 0°.*
- 엉덩이 굴곡: 무릎, 엉덩이, 어깨 사이.\
  *몸통이 허벅지와 일직선일 때 0°.*
- 어깨 굴곡: 엉덩이, 어깨, 팔꿈치 사이.\
  *팔이 몸통과 일직선일 때 180°.*
- 팔꿈치 굴곡: 손목, 팔꿈치, 어깨 사이.\
  *전완이 상완과 일직선일 때 0°.*

**세그먼트 각도 규약:**\
각도는 수평선과 세그먼트 사이에서 반시계 방향으로 측정됩니다.
- 발: 발뒤꿈치와 엄지발가락 사이
- 정강이: 발목과 무릎 사이
- 허벅지: 엉덩이와 무릎 사이
- 골반: 좌우 엉덩이 사이
- 몸통: 엉덩이 중점과 어깨 중점 사이
- 어깨: 좌우 어깨 사이
- 머리: 목과 머리 꼭대기 사이
- 상완: 어깨와 팔꿈치 사이
- 전완: 팔꿈치와 손목 사이


<img src="Content/joint_convention.png" width="760">

<br>

## 인용 방법 및 기여 방법

### 인용 방법
Sports2D를 사용하신다면 [Pagnon, 2024](https://joss.theoj.org/papers/10.21105/joss.06849)를 인용해 주세요.

     @article{Pagnon_Sports2D_Compute_2D_2024,
       author = {Pagnon, David and Kim, HunMin},
       doi = {10.21105/joss.06849},
       journal = {Journal of Open Source Software},
       month = sep,
       number = {101},
       pages = {6849},
       title = {{Sports2D: Compute 2D human pose and angles from a video or a webcam}},
       url = {https://joss.theoj.org/papers/10.21105/joss.06849},
       volume = {9},
       year = {2024}
     }


### 기여 방법
새로운 기능, 코드 개선 등에 대한 제안을 기꺼이 환영합니다!\
Sports2D 또는 Pose2Sim에 기여하고 싶다면 [이 이슈](https://github.com/perfanalytics/pose2sim/issues/40)를 참조하세요.\
할 일 목록이 제안되지만 자유롭게 자신만의 아이디어와 개선 사항을 제안해 주세요.

*할 일 목록: 자유롭게 완성해 주세요:*
- [x] **세그먼트 각도** 계산.
- [x] **다중 사람** 검출, 시간에 따라 일관성 유지.
- [x] **작은 갭만 보간**.
- [x] **필터링 및 플롯 도구**.
- [x] 갑작스러운 **방향 변화** 처리.
- [x] 여러 비디오를 한 번에 분석하기 위한 **배치 처리**.
- [x] 한 사람만 저장하는 옵션 (가장 높은 평균 점수 또는 가장 많은 프레임과 가장 빠른 속도)
- [x] `--load_trc_px` 옵션으로 px .trc 파일에서 포즈 추정 없이 다시 실행.
- [x] 사람 키, 캘리브레이션 파일, 또는 [이미지에서 클릭할](https://stackoverflow.com/questions/74248955/how-to-display-the-coordinates-of-the-points-clicked-on-the-image-in-google-cola) 3D 포인트를 제공하여 **위치를 미터로 변환**
- [x] 모든 검출 및/또는 포즈 추정 모델 지원.
- [x] 선택적으로 사용자가 관심 있는 사람을 선택하게 함.
- [x] OpenSim으로 **역기구학 및 동역학** 수행 (cf. [Pose2Sim](https://github.com/perfanalytics/pose2sim), 하지만 2D로). [이 모델](https://github.com/davidpagnon/Sports2D/blob/main/Sports2D/Utilities/2D_gait.osim) 업데이트 (팔, 마커 추가, 근육 및 접촉 구 제거). 파이프라인 예시 추가.

- [ ] `--compare_to` 옵션으로 실행하여 trc 파일과 모션을 시각적으로 비교. 웹캠 입력으로 실행하면 사용자가 trc 파일의 모션을 따라할 수 있음. 추가 계산으로 특정 변수를 비교할 수 있음.
- [ ] **Colab 버전**: 더 사용자 친화적, 스마트폰에서 사용 가능.
- [ ] Windows, Mac, Linux용, 그리고 Android 및 iOS용 **GUI 애플리케이션**.

</br>

- [ ] 기존 추적 방법(cf. [Kinovea](https://www.kinovea.org/features.html)) 또는 모델 훈련(cf. [DeepLabCut](https://deeplabcut.github.io/DeepLabCut/README.html))으로 **다른 포인트와 각도 추적**.
- [ ] **포즈 정제**. 잘못 추정된 2D 포인트를 클릭하고 이동. 영감을 위해 [DeepLabCut](https://www.youtube.com/watch?v=bEuBKB7eqmk) 참조.
- [ ] 이미지 주석, 왜곡 제거, 원근 고려 등을 위한 도구 추가 (cf. [Kinovea](https://www.kinovea.org/features.html)).


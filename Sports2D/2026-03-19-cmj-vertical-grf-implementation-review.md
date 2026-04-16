# CMJ Vertical GRF Implementation Review

## Scope

이 문서는 현재 코드베이스에서 CMJ(Countermovement Jump)의 vertical GRF를 어떻게 계산하는지, 실제 실행 경로와 수치 처리 순서를 기준으로 정리한 문서다. 코드 자체를 길게 인용하지 않고, 로직과 수식 중심으로 설명한다.

핵심 결론부터 말하면, 이 프로젝트의 CMJ vertical GRF는 force plate 데이터로 측정한 값이 아니라 OpenSim BodyKinematics에서 얻은 전신 CoM(center of mass) 궤적을 시간 미분해서 추정한 값이다. 즉, 본질적으로는 "질량 x 수직가속도 + 체중" 방식의 kinematics-derived estimate다.

## Primary Locations

- `MStudio.NET/src/MStudio.App/ViewModels/MainViewModel.cs`
  - CMJ UI에서 "Estimate GRF" 옵션을 켰을 때 실제로 실행되는 주 경로다.
  - BodyKinematics CSV를 읽고, OpenSim CoM으로 phase를 다시 잡은 뒤, Python GRF estimator를 호출한다.
- `MStudio.NET/src/MStudio.Services/Implementations/Pose2SimWrapperService.cs`
  - C#에서 Python wrapper를 호출하고 JSON 결과를 `GRFEstimationResult`로 역직렬화한다.
- `MStudio.NET/scripts/pose2sim_wrapper.py`
  - 실제 vertical GRF 수식, 필터링, 이중 미분, peak/impulse/RFD 계산이 들어 있는 핵심 구현이다.
- `MStudio.NET/scripts/pose2sim_wrapper.py`의 `run_body_kinematics`
  - OpenSim의 `calcMassCenterPosition`을 이용해 CoM CSV를 만든다. GRF estimator는 이 CSV의 `COM_y`를 입력으로 사용한다.
- `MStudio.NET/src/MStudio.Services/Implementations/CMJPhaseDetectionService.cs`
  - take-off, landing, lowest CoM 같은 이벤트 프레임을 결정한다. 이 프레임들은 GRF clean-up과 impulse 구간 정의에 직접 사용된다.
- `MStudio.NET/src/MStudio.Services/Implementations/Legacy_CMJAnalysisService.cs`
  - 서비스 레이어 차원의 대체 경로다. 다만 현재 App UI의 CMJ "Estimate GRF" 흐름은 이 경로보다 `MainViewModel`의 직접 호출 경로를 더 강하게 사용한다.
- `MStudio.NET/scripts/debug_grf_analysis.py`
  - 운영 경로는 아니고, 필터 cutoff를 바꿔 가며 GRF/impulse를 비교하는 디버그 도구다.

## What Actually Runs in the CMJ UI

현재 App 레이어에서 사용자가 CMJ 분석 창에서 `Estimate GRF`를 선택하면 흐름은 다음과 같다.

1. 기본 CMJ 분석을 먼저 실행한다.
   - 이 단계에서 마커 기반 CoM 또는 기존 phase 정보가 먼저 계산된다.
2. 사용자가 고른 BodyKinematics CSV를 읽는다.
   - 이 CSV에는 OpenSim BodyKinematics로 계산한 `times`, `COM_x`, `COM_y`, `COM_z`, `COP_*` 등이 들어 있다.
3. OpenSim CoM으로 CMJ phase를 다시 검출한다.
   - take-off, landing, lowest CoM, movement start, braking start, landing depth를 다시 잡는다.
4. 그 프레임들을 `EstimateGRFAsync`에 넘긴다.
   - 특히 take-off, landing, lowest CoM을 넘겨서 GRF 곡선 clean-up과 impulse 계산 구간을 정한다.
5. Python에서 vertical GRF 시계열과 discrete metrics를 계산한다.
6. C#에서 결과를 다시 받아 CMJ 결과 객체에 저장한다.
7. 이후 export나 inverse dynamics가 필요하면 이 GRF를 downstream input으로 사용한다.

중요한 점은, 현재 UI 기준 "실제 사용 경로"는 GRF를 서비스 내부에서 자동으로 다 끝내는 형태가 아니라, `MainViewModel`이 OpenSim CoM 재검출과 GRF 추정을 단계적으로 조합하는 형태라는 점이다.

## CoM Source Used for GRF

vertical GRF 추정의 입력은 TRC 마커 좌표가 아니라 OpenSim BodyKinematics 결과의 CoM 시계열이다.

BodyKinematics 단계에서는 각 프레임에서 OpenSim 모델 상태를 재현한 뒤 전신 CoM를 계산한다. 그 결과가 CSV로 저장되며, GRF estimator는 여기서 다음 두 열만 필수적으로 사용한다.

- 시간축: `times` 또는 `time`
- 수직 CoM: `COM_y`

즉, CMJ GRF 추정은 "전신 CoM의 수직 위치 곡선"만 있으면 돌아간다.

## Coordinate Assumption

현재 GRF estimator는 수직축을 항상 `COM_y`라고 가정한다.

이 가정이 성립하는 이유는 현재 C# 쪽 `RunBodyKinematicsAsync` 호출이 기본값인 `direction = "yup"`를 사용하기 때문이다. 이 경우 OpenSim의 y-up 좌표계에서 `COM_y`가 실제 수직축이다.

주의할 점은, `run_body_kinematics`는 `zup` 변환도 지원하지만, GRF estimator는 여전히 `COM_y`를 읽는다는 점이다. 따라서 BodyKinematics CSV를 `zup`으로 만들면 vertical axis와 estimator가 어긋날 수 있다. 현재 CMJ UI 경로에서는 기본 `yup`를 쓰므로 이 문제는 직접 발생하지 않는다.

## Phase/Event Frames That Affect GRF

GRF 자체는 CoM 미분으로 계산되지만, 결과를 어떻게 해석할지는 phase frame에 강하게 의존한다.

### 1. OpenSim CoM으로 phase를 다시 잡음

UI 경로에서는 BodyKinematics CSV에서 읽은 OpenSim CoM를 이용해 `CMJPhaseDetectionService`를 다시 돌린다. 이 결과로 얻는 값은 다음과 같다.

- movement start
- braking start
- lowest CoM
- take-off
- peak height
- landing
- landing depth

### 2. 실제 구현상 take-off/landing 판정 방식

이 서비스는 문서나 주석만 보면 toe height 기반 판정처럼 보일 수 있지만, 현재 활성 코드의 핵심 판정은 더 다르다.

- movement start:
  - CoM 수직속도의 최솟값(가장 빠른 하강 속도)의 5%를 넘는 첫 하강 프레임
- braking start:
  - movement start 이후 첫 속도 최소값
- lowest CoM:
  - pre-takeoff 구간에서 CoM 높이 최소값
- max velocity:
  - lowest CoM 이후 CoM 수직속도 최대값
- take-off:
  - 현재 구현은 toe lift를 직접 쓰지 않고, `max velocity + 1 frame`을 사실상 take-off로 사용한다
- landing:
  - peak height 이후 하강 구간에서 "가장 큰 음의 속도"가 나타나는 프레임을 초기 접지로 본다
- landing depth:
  - landing 이후 두 번째 CoM 최소값

즉, GRF 추정에 전달되는 take-off/landing 프레임은 현재 구현 기준으로는 toe threshold보다 velocity-derived event에 더 가깝다.

### 3. Toe 기반 로직은 보조적 성격

toe height 기반 take-off/landing 함수가 코드에 남아 있기는 하지만, 현재 핵심 detect flow에서는 직접 사용되지 않는다. toe position은 보조 정보나 기록용 성격이 더 강하다.

## Vertical GRF Time Series Calculation

실제 vertical GRF 계산은 Python `estimate_grf_from_com`에서 수행된다. 계산 순서는 아래와 같다.

### Step 1. 시간 간격과 샘플링 주파수 계산

BodyKinematics CSV에서 시간열을 읽고

- `dt = t[i+1] - t[i]`
- `frame_rate = 1 / dt`

로 계산한다.

### Step 2. CoM 수직 위치 low-pass filtering

`COM_y`에 4차 Butterworth low-pass filter를 적용한다.

- cutoff frequency: 12 Hz
- filter order: 4
- 적용 방식: zero-phase `filtfilt`

Nyquist보다 12 Hz가 크면 필터를 생략한다.

즉, 실제 미분에 쓰이는 CoM은 raw `COM_y`가 아니라 `COM_y_filtered`다.

### Step 3. 위치 -> 속도 -> 가속도 이중 미분

NumPy `gradient`를 두 번 적용한다.

- `v_y = d(COM_y_filtered) / dt`
- `a_y = d(v_y) / dt`

중간 프레임은 중앙차분에 가깝고, 양 끝 프레임은 단측 차분에 가깝게 처리된다.

### Step 4. vertical GRF 계산

중력가속도 `g = 9.81 m/s^2`를 사용해 각 프레임의 vertical GRF를 다음과 같이 계산한다.

`vGRF[i] = m * a_y[i] + m * g`

여기서

- `m`은 체중(kg)
- `a_y`는 CoM 수직가속도
- `m * g`는 정적 체중(body weight)

이다.

따라서 정지 상태에서 `a_y = 0`이면 `vGRF = body weight`가 된다.

## How Take-off and Landing Affect the GRF Curve

GRF 곡선을 만든 다음, phase frame을 이용해 비행 구간을 강제로 0으로 만든다.

### Take-off frame

우선순위는 다음과 같다.

1. C#에서 넘긴 `forced_takeoff`
2. 없으면 fallback으로 GRF가 `0.1 * body weight` 이하로 떨어지는 첫 시점

즉, UI 경로에서는 보통 phase detector에서 계산한 take-off 프레임이 그대로 사용된다.

### Landing frame

우선순위는 다음과 같다.

1. C#에서 넘긴 `forced_landing`
2. 없으면 fallback으로 take-off 이후 5프레임 지난 뒤, GRF가 `0.5 * body weight`를 넘는 첫 시점

### Flight phase clean-up

`takeoff + 1`부터 `landing - 1`까지의 GRF는 0으로 강제 설정한다.

즉, 계산식 자체는 CoM 미분으로 만들어진 연속 곡선이지만, 비행 구간만큼은 "공중에서는 지면반력이 없어야 한다"는 물리 제약을 적용해 후처리한다.

추가로, 현재 `MainViewModel`도 C# 쪽에서 같은 flight 구간을 다시 0으로 덮어쓴다. 따라서 UI 주 경로에서는 비행 구간 zeroing이 Python과 C#에서 한 번씩 중복 적용된다.

## Discrete Metrics Derived from the GRF Curve

### Body Weight

`BW = m * g`

이 값은 peak normalization, threshold 판정, impulse 계산의 기준선으로 사용된다.

### Peak Vertical GRF

peak GRF는 take-off 이전 구간에서 최대값으로 계산된다.

- 검색 구간: 첫 프레임부터 take-off 직전까지
- 계산식: `peak_vGRF = max(vGRF[0 : takeoff])`

즉, propulsion 중 최대 수직 힘을 의미한다.

### Net Vertical Impulse

net impulse는 "GRF - body weight"를 적분한 값이다.

기본 수식은 다음과 같다.

`Net Impulse = integral( vGRF(t) - BW ) dt`

실제 적분 구간은 propulsion start부터 take-off까지다.

propulsion start는 다음 우선순위로 결정된다.

1. `forced_lowest_com`가 전달되면 그 프레임
2. 그렇지 않으면 GRF가 체중선(BW)을 상향 돌파하는 첫 시점에 가까운 프레임을 fallback으로 사용

적분 자체는 trapezoidal rule(`trapz`)로 수행한다.

즉, 현재 구현에서 net impulse는 "lowest CoM 이후 take-off 전까지 체중을 초과하거나 미달하는 힘의 순합"으로 정의된다.

### RFD (Rate of Force Development)

RFD는 peak 직전까지의 GRF 중 최솟값과 peak 사이의 기울기로 계산된다.

`RFD = (vGRF_peak - vGRF_min_before_peak) / delta_t`

여기서 `vGRF_min_before_peak`는 peak 이전 전 구간의 전역 최솟값에 가깝다. 따라서 biomechanics 문헌에서 자주 쓰는 "force onset 이후 특정 구간의 RFD"와 완전히 동일한 정의는 아니다.

## Differences Between the Two CMJ GRF Paths

코드베이스에는 CMJ GRF 관련 경로가 두 개 있다.

### Path A. 현재 UI에서 더 직접적으로 쓰이는 경로

`MainViewModel`이

1. BodyKinematics CSV 로드
2. OpenSim CoM 기반 phase 재검출
3. `EstimateGRFAsync(bodyKinCsv, mass, takeoff, landing, lowestCoM)` 호출

을 수행한다.

이 경로에서는 lowest CoM도 Python에 전달되므로 net impulse 구간이 phase detector 결과와 잘 맞는다.

### Path B. 서비스 레이어의 대체 경로

`Legacy_CMJAnalysisService` 내부에도 OpenSim GRF 실행 함수가 있다. 이 경로는

- take-off
- landing

은 넘기지만, `lowestCoMFrame`은 넘기지 않는다.

그 결과 이 경로가 사용될 경우 net impulse의 propulsion start는 Python fallback 로직에 의해 정해진다. 따라서 Path A와 impulse 값이 달라질 수 있다.

## Downstream Use of the Estimated Vertical GRF

계산된 vertical GRF는 단순 표시용으로만 쓰이지 않는다.

### 1. CMJ 결과 및 그래프

결과 객체에 다음 값이 저장된다.

- peak vertical GRF
- net vertical impulse
- RFD
- 전체 GRF time series

이 값들은 결과창, export, 그래프에 그대로 사용된다.

### 2. OpenSim External Loads 생성

Inverse Dynamics를 돌릴 때는 추정된 total vertical GRF를 좌우 발에 50:50으로 나눠 `.mot` 파일을 만든다.

즉, downstream에서는 다음과 같은 단순화가 적용된다.

- 전체 vGRF를 양발에 절반씩 분배
- 수평 힘 성분은 0으로 둠
- 좌우 모두 같은 CoP를 사용

이는 bilateral CMJ를 가정한 단순화 모델이지, 좌우 개별 force plate를 복원하는 모델은 아니다.

## Related Files and Their Roles

- `MStudio.NET/src/MStudio.Core/Interfaces/IPose2SimService.cs`
  - C# 인터페이스 수준에서도 GRF가 `GRF_y = m * a_y + m * g`라는 가정 위에 있음을 명시한다.
- `MStudio.NET/src/MStudio.Core/Models/Analysis/GRFEstimationResult.cs`
  - Python에서 돌아오는 결과를 저장하는 모델이다.
  - 현재 모델에는 take-off는 있지만 landing 필드는 없다.
- `MStudio.NET/src/MStudio.Core/Models/Analysis/CMJAnalysisResult.cs`
  - peak GRF, impulse, RFD, GRF time series를 최종 CMJ 결과에 보관한다.
- `MStudio.NET/scripts/debug_grf_analysis.py`
  - 6 Hz, 12 Hz, 20 Hz 필터 비교와 impulse sanity check를 하는 검증 도구다.

## Important Implementation Notes

### 1. 이 구현은 measured GRF가 아니라 estimated GRF다

force plate를 읽는 것이 아니라 CoM 이중 미분으로 추정한다. 따라서 결과는 모델링, filtering, phase alignment 품질에 영향을 크게 받는다.

### 2. vertical axis는 사실상 `COM_y`에 고정돼 있다

현재 운영 경로에서는 문제가 없지만, `zup` CSV를 같은 estimator에 넣으면 수직축 해석이 틀어질 수 있다.

### 3. take-off와 landing은 현재 velocity-derived 성격이 강하다

주석이나 명명만 보면 toe-based처럼 보이지만, 실제 활성 detect flow는 `max velocity + 1 frame`과 peak negative velocity를 더 강하게 사용한다.

### 4. 비행 구간 0 처리 로직이 두 번 있다

Python에서 한 번, `MainViewModel`에서 한 번 더 적용한다. 결과적으로 UI 경로에서는 flight phase가 매우 강하게 0으로 고정된다.

### 5. net impulse는 경로에 따라 달라질 수 있다

`lowestCoMFrame`을 Python에 전달하는지 여부에 따라 propulsion start가 바뀌기 때문이다. 현재 UI 직접 경로는 전달하고, `Legacy_CMJAnalysisService` 경로는 전달하지 않는다.

### 6. RFD 정의는 비교적 단순하다

force onset detection 없이 "peak 이전 최솟값 -> peak" 구간의 평균 기울기를 사용한다. 그래서 연구용 force plate 파이프라인의 정교한 RFD 정의와는 다를 수 있다.

### 7. landing frame은 Python JSON에는 있지만 C# typed metric에는 완전히 실리지 않는다

Python 결과에는 landing frame이 포함되지만, `GRFMetrics`는 현재 take-off만 저장한다. UI 경로에서는 별도로 phase detector가 landing을 이미 갖고 있어서 기능상 큰 문제는 없지만, 데이터 모델은 비대칭적이다.

## Bottom Line

현재 코드베이스의 CMJ vertical GRF는 다음 한 줄로 요약할 수 있다.

1. OpenSim BodyKinematics로 전신 CoM의 `COM_y`를 얻는다.
2. `COM_y`를 12 Hz Butterworth로 필터링한다.
3. 이를 두 번 미분해 수직가속도를 얻는다.
4. `vGRF = m * a_y + m * g`로 수직 GRF를 계산한다.
5. take-off, landing, lowest CoM 프레임으로 비행 구간과 propulsion 구간을 정리한다.
6. 그 위에서 peak GRF, net impulse, RFD를 계산한다.

즉, 이 프로젝트의 CMJ vGRF 구현은 "OpenSim CoM 기반의 kinematics-derived vGRF estimator"라고 이해하면 가장 정확하다.

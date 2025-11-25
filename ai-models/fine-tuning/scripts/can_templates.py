#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAN Data to Korean Conversation Templates
Based on qwen3-truck-korean_spec.yaml patterns
"""

import random
from typing import Dict, List, Any

class CANConversationTemplates:
    """10 CAN data patterns with Korean conversation templates"""

    def __init__(self):
        self.templates = {
            'speeding': self._speeding_templates,
            'harsh_braking': self._harsh_braking_templates,
            'high_rpm': self._high_rpm_templates,
            'full_throttle': self._full_throttle_templates,
            'city_driving': self._city_driving_templates,
            'highway_cruising': self._highway_cruising_templates,
            'idle_stopped': self._idle_stopped_templates,
            'acceleration': self._acceleration_templates,
            'deceleration': self._deceleration_templates,
            'normal_driving': self._normal_driving_templates,
        }

    def _speeding_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """과속 (>100 km/h)"""
        speed = data.get('vehicle_speed', 110)
        rpm = data.get('engine_rpm', 2800)

        templates = [
            {
                'instruction': '지금 차량 상태가 어떤가요?',
                'response': f'현재 속도 {speed:.0f}km/h로 주행 중입니다. 과속 중이니 속도를 줄이시는 것이 안전합니다. 엔진 RPM은 {rpm:.0f}입니다.'
            },
            {
                'instruction': '속도가 너무 빠른 것 같은데 괜찮나요?',
                'response': f'네, {speed:.0f}km/h는 과속입니다. 안전을 위해 100km/h 이하로 감속하시고, 앞차와의 안전거리를 확보하세요.'
            },
            {
                'instruction': '과속 경고가 떴어요',
                'response': f'현재 속도가 {speed:.0f}km/h입니다. 제한속도를 초과했습니다. 속도를 줄이고 주변 차량과 안전거리를 유지하세요.'
            },
        ]
        return templates

    def _harsh_braking_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """급제동 (brake_pressure > 80)"""
        brake = data.get('brake_pressure', 95)
        speed_before = data.get('vehicle_speed', 80)
        speed_after = max(0, speed_before - 30)

        templates = [
            {
                'instruction': '방금 급제동 경고가 떴는데 무슨 일인가요?',
                'response': f'급제동이 감지되었습니다. 브레이크 압력 {brake:.0f}%, 속도가 {speed_before:.0f}km/h에서 {speed_after:.0f}km/h로 급감했습니다. 앞차와의 거리를 충분히 확보하세요.'
            },
            {
                'instruction': '왜 급제동 알림이 나오죠?',
                'response': f'브레이크를 {brake:.0f}% 압력으로 밟아 급격히 감속했습니다. 안전거리 유지와 예측 운전으로 급제동을 피하시는 게 좋습니다.'
            },
            {
                'instruction': '급제동을 피하려면 어떻게 해야 하나요?',
                'response': '앞차와의 안전거리를 2초 이상 유지하고, 교통 흐름을 미리 예측하여 부드럽게 감속하세요. 도로 상황을 넓게 보는 것이 중요합니다.'
            },
        ]
        return templates

    def _high_rpm_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """고RPM (>3000)"""
        rpm = data.get('engine_rpm', 3500)
        speed = data.get('vehicle_speed', 70)

        templates = [
            {
                'instruction': '엔진 소리가 크게 들리는데 이상한가요?',
                'response': f'엔진 RPM이 {rpm:.0f}으로 높습니다. 현재 속도 {speed:.0f}km/h에 비해 기어가 낮은 것 같습니다. 변속을 올려 연비를 개선하세요.'
            },
            {
                'instruction': 'RPM이 높다는 경고가 나와요',
                'response': f'RPM {rpm:.0f}은 다소 높은 수치입니다. 기어를 높이거나 가속 페달을 조금 줄여 2000-2500 RPM으로 유지하는 것이 연비에 좋습니다.'
            },
            {
                'instruction': '연비를 개선하려면 어떻게 하죠?',
                'response': f'현재 RPM {rpm:.0f}을 2000-2500으로 낮추세요. 적절한 기어 선택과 부드러운 가속이 연비 개선의 핵심입니다.'
            },
        ]
        return templates

    def _full_throttle_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """풀스로틀 (>85%)"""
        throttle = data.get('throttle_position', 92)
        speed = data.get('vehicle_speed', 65)

        templates = [
            {
                'instruction': '가속이 너무 급한 것 같은데요',
                'response': f'스로틀 {throttle:.0f}%로 급가속 중입니다. 부드러운 가속이 연비와 안전에 좋습니다. 스로틀을 70% 이하로 조절하세요.'
            },
            {
                'instruction': '연료가 빨리 줄어드는 이유가 뭔가요?',
                'response': f'스로틀 {throttle:.0f}%의 급가속이 주요 원인입니다. 부드럽게 가속하면 연비를 20-30% 개선할 수 있습니다.'
            },
            {
                'instruction': '안전 운전 팁 좀 알려주세요',
                'response': '급가속을 피하고 스로틀을 50-70%로 유지하세요. 예측 운전으로 불필요한 가감속을 줄이면 연비와 안전성이 향상됩니다.'
            },
        ]
        return templates

    def _city_driving_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """시내 주행 (<30 km/h)"""
        speed = data.get('vehicle_speed', 25)
        rpm = data.get('engine_rpm', 1200)

        templates = [
            {
                'instruction': '시내 주행 중인데 연비가 너무 안 나와요',
                'response': f'현재 평균 속도 {speed:.0f}km/h의 시내 주행은 정체로 인해 연비가 낮습니다. 출퇴근 시간을 피하거나 우회 도로를 이용하면 개선됩니다.'
            },
            {
                'instruction': '저속 주행 시 주의사항이 있나요?',
                'response': f'속도 {speed:.0f}km/h 시내 주행에서는 급정거, 급가속을 피하고 앞차와의 거리를 충분히 유지하세요. 보행자와 이륜차에 특히 주의하세요.'
            },
            {
                'instruction': '시내에서 연비 개선 방법은?',
                'response': '불필요한 공회전을 줄이고, 신호를 미리 예측하여 부드럽게 가감속하세요. 에코 드라이브 모드를 활성화하는 것도 도움됩니다.'
            },
        ]
        return templates

    def _highway_cruising_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """고속도로 순항 (80-100 km/h)"""
        speed = data.get('vehicle_speed', 90)
        rpm = data.get('engine_rpm', 2200)

        templates = [
            {
                'instruction': '고속도로에서 최적 속도가 어떻게 되나요?',
                'response': f'현재 {speed:.0f}km/h는 연비와 안전을 고려할 때 최적입니다. 이 속도를 일정하게 유지하면 연비가 가장 좋습니다.'
            },
            {
                'instruction': '장거리 운행 팁 좀 주세요',
                'response': f'{speed:.0f}km/h로 일정한 속도를 유지하고, 2시간마다 휴게소에서 휴식하세요. 크루즈 컨트롤을 사용하면 피로도 감소에 도움됩니다.'
            },
            {
                'instruction': '고속도로 주행이 효율적인가요?',
                'response': f'네, 현재 {speed:.0f}km/h 정속 주행은 시내 주행보다 30-40% 연비가 좋습니다. 급가감속을 피하고 일정 속도를 유지하세요.'
            },
        ]
        return templates

    def _idle_stopped_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """정차/공회전 (speed == 0)"""
        rpm = data.get('engine_rpm', 800)

        templates = [
            {
                'instruction': '공회전 중인데 시동을 끄는 게 나을까요?',
                'response': f'1분 이상 정차 시 시동을 끄는 것이 연료 절약에 좋습니다. 현재 RPM {rpm:.0f}으로 공회전 중입니다.'
            },
            {
                'instruction': '공회전이 연비에 어떤 영향을 주나요?',
                'response': '10분 공회전은 약 200-300ml의 연료를 소비합니다. 신호 대기나 하역 중에는 시동을 끄는 것을 권장합니다.'
            },
            {
                'instruction': '정차 시 주의사항은?',
                'response': '경사로에서는 주차 브레이크를 확실히 걸고, 1분 이상 정차 시 시동을 끄세요. 겨울철에는 워밍업 시간을 30초 정도만 주세요.'
            },
        ]
        return templates

    def _acceleration_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """가속 중 (throttle increasing)"""
        throttle = data.get('throttle_position', 65)
        speed = data.get('vehicle_speed', 55)

        templates = [
            {
                'instruction': '가속 중인데 올바른 방법인가요?',
                'response': f'스로틀 {throttle:.0f}%는 적절한 수준입니다. 50-70% 범위에서 부드럽게 가속하면 연비와 안전성이 좋습니다.'
            },
            {
                'instruction': '합류 구간에서 가속 요령이 있나요?',
                'response': f'현재 {speed:.0f}km/h에서 본선 속도에 맞춰 가속하세요. 스로틀 70-80%로 충분한 가속력을 확보하면서 안전하게 합류하세요.'
            },
            {
                'instruction': '부드러운 가속이 중요한가요?',
                'response': f'네, 급가속 대비 부드러운 가속은 연비를 20-30% 개선합니다. 스로틀 {throttle:.0f}%를 유지하면서 점진적으로 속도를 올리세요.'
            },
        ]
        return templates

    def _deceleration_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """감속 중 (brake + speed decreasing)"""
        brake = data.get('brake_pressure', 45)
        speed = data.get('vehicle_speed', 60)

        templates = [
            {
                'instruction': '안전하게 감속하는 방법은?',
                'response': f'현재 브레이크 압력 {brake:.0f}%는 적절합니다. 급제동을 피하고 엔진 브레이크를 먼저 사용하면 더 안전합니다.'
            },
            {
                'instruction': '내리막길에서 감속 요령이 있나요?',
                'response': f'{speed:.0f}km/h에서 엔진 브레이크를 활용하세요. 기어를 1-2단 낮추면 브레이크 과열을 방지하고 안전하게 감속할 수 있습니다.'
            },
            {
                'instruction': '브레이크를 덜 쓰는 방법은?',
                'response': '신호와 교통 흐름을 미리 예측하여 가속 페달만 놓고 자연 감속하세요. 엔진 브레이크를 먼저 사용하면 브레이크 수명이 늘어납니다.'
            },
        ]
        return templates

    def _normal_driving_templates(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """정상 주행 (50-80 km/h)"""
        speed = data.get('vehicle_speed', 65)
        rpm = data.get('engine_rpm', 2000)
        throttle = data.get('throttle_position', 50)

        templates = [
            {
                'instruction': '현재 운전 상태가 괜찮은가요?',
                'response': f'매우 좋습니다. 속도 {speed:.0f}km/h, RPM {rpm:.0f}, 스로틀 {throttle:.0f}%는 모두 최적 범위입니다. 이 상태를 유지하세요.'
            },
            {
                'instruction': '안전 운전과 연비를 동시에 잡으려면?',
                'response': f'현재처럼 {speed:.0f}km/h 정속 주행, RPM 2000-2500 유지, 부드러운 가감속이 핵심입니다. 잘하고 계십니다.'
            },
            {
                'instruction': '장시간 운전 시 주의사항은?',
                'response': f'현재 {speed:.0f}km/h 안정적 주행을 유지하면서 2시간마다 휴식하세요. 수분 섭취와 스트레칭도 잊지 마세요.'
            },
        ]
        return templates

    def generate(self, pattern: str, can_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate conversation from CAN data

        Args:
            pattern: One of 10 patterns (speeding, harsh_braking, etc.)
            can_data: CAN data dict with keys like vehicle_speed, engine_rpm, etc.

        Returns:
            Dict with 'instruction' and 'response' keys
        """
        if pattern not in self.templates:
            raise ValueError(f"Unknown pattern: {pattern}")

        # Get templates for this pattern
        templates = self.templates[pattern](can_data)

        # Randomly select one
        return random.choice(templates)

    def get_pattern_for_data(self, can_data: Dict[str, Any]) -> str:
        """Detect pattern from CAN data"""
        speed = can_data.get('vehicle_speed', 0)
        rpm = can_data.get('engine_rpm', 0)
        throttle = can_data.get('throttle_position', 0)
        brake = can_data.get('brake_pressure', 0)

        # Priority order (most specific first)
        if speed > 100:
            return 'speeding'
        elif brake > 80:
            return 'harsh_braking'
        elif rpm > 3000:
            return 'high_rpm'
        elif throttle > 85:
            return 'full_throttle'
        elif speed == 0:
            return 'idle_stopped'
        elif speed < 30:
            return 'city_driving'
        elif 80 <= speed <= 100:
            return 'highway_cruising'
        elif brake > 30:  # Decelerating
            return 'deceleration'
        elif throttle > 60:  # Accelerating
            return 'acceleration'
        else:  # 50-80 km/h, normal params
            return 'normal_driving'


if __name__ == '__main__':
    # Test templates
    templates = CANConversationTemplates()

    # Test cases
    test_cases = [
        ('speeding', {'vehicle_speed': 115, 'engine_rpm': 3200, 'throttle_position': 75}),
        ('harsh_braking', {'vehicle_speed': 80, 'brake_pressure': 95}),
        ('high_rpm', {'vehicle_speed': 70, 'engine_rpm': 3500}),
        ('normal_driving', {'vehicle_speed': 65, 'engine_rpm': 2100, 'throttle_position': 55}),
    ]

    print('[CAN CONVERSATION TEMPLATE TEST]')
    print()

    for pattern, data in test_cases:
        conv = templates.generate(pattern, data)
        print(f'Pattern: {pattern}')
        print(f'  CAN Data: {data}')
        print(f'  User: {conv["instruction"]}')
        print(f'  Assistant: {conv["response"]}')
        print()
